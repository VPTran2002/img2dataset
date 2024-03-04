from torch.utils.data import Dataset, DataLoader
import fsspec
import os
import torch
import open_clip
import pyarrow.parquet as pq
import pyarrow as pa
import heapq
import time
import pickle
import requests
import cv2
import numpy as np
import albumentations as A
from multiprocessing.pool import ThreadPool
from threading import Semaphore
from writer import WebDatasetSampleWriter


def resize_img(img):
    cv2.setNumThreads(1)
    img_size=256
    original_height, original_width = img.shape[:2]
    downscale = max(original_width, original_height) > img_size
    interpolation = cv2.INTER_AREA if downscale else cv2.INTER_LANCZOS4
    img = A.longest_max_size(img, img_size, interpolation=interpolation)
    img = A.pad(
        img,
        img_size,
        img_size,
        border_mode=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    return img

def download_resize(dist_url_caption, timeout, semaphore):
    try:
        # Download image
        user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
        headers={"User-Agent": user_agent_string}
        print(dist_url_caption[1])
        response = requests.get(dist_url_caption[1], timeout=timeout, headers=headers)
        if response.status_code == 200:
            # Decode image
            img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
            # Resize image
            resized_img = resize_img(img)
            #write image to destination
            encode_format = "jpg"
            img_str = cv2.imencode(f".{encode_format}", resized_img, params=[int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
            semaphore.release()
            return img_str, dist_url_caption[2]
        else:
            semaphore.release()
            print(f"Failed to download image from {dist_url_caption[1]}. Status code: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        semaphore.release()
        return None, None


class DatasetMetaData(Dataset):
    def __init__(self, dataframe, tokenizer, batch_size):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.total_rows = len(dataframe)
        self.num_batches = (self.total_rows + self.batch_size - 1) // self.batch_size
        self.batch_start = -1
        self.current_slice = self.dataframe.to_pandas()

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        caption = self.current_slice.iloc[idx]['caption']
        url = self.current_slice.iloc[idx]['url']
        if caption is None:
            return "", url
        return caption, url 

class Laion400mDataset(Dataset):
    def __initialize_model(self, model, pretrained):
        self.model, _, _ = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model)
        with torch.no_grad():
            self.caption_features_targets = self.model.encode_text(self.tokenizer(self.captions).to(self.device)) #this is a matrix of dimension Nx(dimension feature embedding)
            self.caption_features_targets /= self.caption_features_targets.norm(dim=-1, keepdim=True)

    def __initialize_priority_queue(self):
        prqueue = self.__make_path_absolute(self.priority_queue_save_path)
        fs, prqueue_path = fsspec.core.url_to_fs(prqueue)
        file_path_queue = f"{prqueue_path}/queue.pkl"
        if fs.exists(file_path_queue):
            with fs.open(file_path_queue, 'rb') as f:
                self.priority_queues = pickle.load(f)
        else:
            self.priority_queues = []
            for i in range(len(self.captions)):
                self.priority_queues.append([])

    def __make_path_absolute(self, path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    def __get_names_meta_data_files(self, pathToMeta):
        pathToMeta = self.__make_path_absolute(pathToMeta)
        fs, url_path = fsspec.core.url_to_fs(pathToMeta)
        self.fs = fs
        max = self.__get_max_input_file_filtered()
        if fs.isdir(url_path):
            input_files = sorted(fs.glob(url_path + "/*." + self.input_format))
            input_files = self.__filter_out_processed_files(input_files, max)
        else:
            input_files = [url_path]

        start_file = max+1
        return input_files, start_file

    def __filter_out_processed_files(self, file_list, max):
        filtered_file_list = []
        for path in file_list:
            name = path.split('/')[-1]
            number = int(name[5:10])
            if number > max:
                 filtered_file_list.append(path)
        return filtered_file_list

    def __get_max_input_file_filtered(self):
        prqueue = self.__make_path_absolute(self.priority_queue_save_path)
        fs_queue, prqueue_path = fsspec.core.url_to_fs(prqueue)
        if not fs_queue.exists(prqueue_path):
            fs_queue.mkdir(prqueue_path)
        files = fs_queue.ls(prqueue_path)
        max = 0
        for file in files:
            name = file.split('/')[-1]
            if name.startswith("status"):
                number_part = int(name[6:])
                if number_part > max:
                    max = number_part
        return max            


    def __create_output_directory(self, output_folder):
        self.output_folder = self.__make_path_absolute(output_folder)
        fs, output_path = fsspec.core.url_to_fs(self.output_folder)
        if not fs.exists(output_path):
            fs.mkdir(output_path)
        

    def __collect_urls(self, meta_data_files: list, start_file: int):
        for i, input_file in enumerate(meta_data_files):
            print("File_number " + str(start_file + i))
            self.__collect_urls_file(meta_data_file=input_file)
            self.__save_priority_queue(start_file + i)

    def __collect_urls_file(self, meta_data_file):
        df = self.__download_meta_to_pyarrow_table(meta_data_file)
        df = self.__rename_cols_in_pyarrow_table(df)
        list_df = self.__divide_dataset_into_1m_shards(df)
        def collate_fn(batch):
            with torch.no_grad():
                caption, url = zip(*batch)
                return self.tokenizer(caption), url        
        for i in range(len(list_df)):
            print("Shard number: " + str(i))
            dataset_url_cap = DatasetMetaData(list_df[i], self.tokenizer, self.batch_size_meta)
            dataloader = DataLoader(dataset_url_cap, batch_size=self.batch_size_meta, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers)#self.num_workers)
            self.__updatePriorityQueue(dataloader)

            
    def __save_priority_queue(self, filenumber):
        self.__create_output_directory(self.priority_queue_save_path)
        prqueue = self.__make_path_absolute(self.priority_queue_save_path)
        #save priority queue
        fs, prqueue_path = fsspec.core.url_to_fs(prqueue)
        file_path_queue = f"{prqueue_path}/queue.pkl"
        with fs.open(file_path_queue, 'wb') as f:
            pickle.dump(self.priority_queues, f)
        #save status
        file_path_status = f"{prqueue_path}/status" + str(filenumber)
        with fs.open(file_path_status, 'w') as f:
            pass
        #delete previous status
        self.__delete_prev_status_files(fs, prqueue_path, filenumber)


    def __delete_prev_status_files(self, fs, prqueue_path, filenumber):
        files = fs.ls(prqueue_path)
        for file in files:
            name = file.split('/')[-1]
            if name.startswith("status"):
                number_part = int(name[6:])
                if number_part < filenumber:
                    fs.rm(file) 

    def __divide_dataset_into_1m_shards(self, df):
        list_datasets = []
        len_dataset = len(df)
        len_shard = (len_dataset + self.shard_size-1)//self.shard_size
        for i in range(len_shard):
            batch_start = i * self.shard_size
            batch_end = min(batch_start + self.shard_size, len_dataset)
            shard = df.slice(batch_start, batch_end-batch_start)
            list_datasets.append(shard)
        return list_datasets    

    def __rename_cols_in_pyarrow_table(self, df):
        column_names = ['url', 'caption']
        df = df.rename_columns(column_names)
        return df    


    def __download_meta_to_pyarrow_table(self, meta_data_file):
        with self.fs.open(meta_data_file, mode="rb") as file:
            df = pq.read_table(file, columns=self.columns_to_read)
        return df

    def __updatePriorityQueue(self, dataloader: DataLoader):
        with torch.no_grad():
            j = 0
            start = time.time()
            for batch in dataloader:
                caption_tokens = batch[0].to(self.device)
                caption_features = self.model.encode_text(caption_tokens)
                caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)
                #calculate distances
                distances = caption_features @ self.caption_features_targets.T
                shortest_distance, index_shortest_distance = distances.min(dim=-1)
                shortest_distance_np = -shortest_distance.to("cpu").numpy() # welche priorität
                index_shortest_distance_np = index_shortest_distance.to("cpu").numpy() #welche priority queue
                urls = batch[1] #welcher value in der priority
                for i in range(shortest_distance_np.shape[0]):
                    if(
                        (len(self.priority_queues[index_shortest_distance_np[i]]) < 
                        self.num_elements_per_caption) or 
                        (shortest_distance_np[i] > 
                        self.priority_queues[index_shortest_distance_np[i]][0][0])):

                        heapq.heappush(self.priority_queues[index_shortest_distance_np[i]], (shortest_distance_np[i], urls[i], self.tokenizer.decode(caption_tokens[i].cpu().numpy())))
                if(j % 50 == 0):
                    print(j)
                j = j + 1            
                self.__cut_down_prriority_queues()

            stop = time.time()
            duration_single_example = (stop-start)/self.shard_size
            print("Duration " + str(duration_single_example))
    

    def __cut_down_prriority_queues(self):
        for i in range(len(self.priority_queues)):
            priority_queue_i = self.priority_queues[i]
            if(len(priority_queue_i) <= self.num_elements_per_caption):
                break
            for j in range(len(priority_queue_i)-self.num_elements_per_caption):
                heapq.heappop(priority_queue_i)


    def __download_urls(self):
        i = 0
        for url_caption_list in self.priority_queues:
            self.__download_class(url_caption_list, i)
            i += 1


    def __download_class(self, url_caption_list, class_number):   
        semaphore = Semaphore(self.thread_count * 2)
        def data_generator():
            for e in url_caption_list:
                semaphore.acquire()  # pylint: disable=consider-using-with
                yield e

        loader = data_generator()

        schema = pa.schema([])
        schema = (
            schema.append(pa.field("caption", pa.string()))
            .append(pa.field("class", pa.string()))
        )
        sample_writer = WebDatasetSampleWriter(
            shard_id=class_number,
            output_folder=self.output_folder,
            save_caption=True,
            oom_shard_count=len(self.captions),
            schema=schema,
            encode_format="jpg"
        )

        with ThreadPool(self.thread_count) as thread_pool:
            for img_str, caption in thread_pool.imap_unordered(lambda dist_url_caption: download_resize(dist_url_caption, self.timeout, semaphore=semaphore),loader):
                try:
                    if img_str is None:
                        continue
                    meta = {
                        'caption': caption,
                        'class': self.captions[class_number]
                    }
                    sample_writer.write(img_str, "None", caption, meta)
                except Exception as e:
                    print(f"Some error occured: {str(e)}")

            sample_writer.close()
            thread_pool.terminate()
            thread_pool.join()
            del thread_pool                                                 

    def __init__(self, num_elements_per_caption: int, batch_size_meta: int, 
                num_workers:int, output_folder: str = "laion400m-data", 
                pathToMeta: str = "laion400m-meta", 
                thread_count: int = 1, image_size: int = 256, timeout: int = 10,
                model: str = 'ViT-B-32', pretrained: str = 'laion400m_e32',
                captions: list = ["a dog", "a cat", "an airplane"],
                shard_size: int = 500000, priority_queue_save_path: str = "prqueue"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.columns_to_read = ["URL", "TEXT"]
        self.input_format = "parquet"
        self.output_format = "webdataset"
        self.captions = captions
        self.num_elements_per_caption = num_elements_per_caption
        self.batch_size_meta = batch_size_meta
        self.num_workers = num_workers
        self.shard_size = shard_size
        self.image_size = image_size
        self.priority_queue_save_path = priority_queue_save_path
        self.thread_count = thread_count
        self.timeout = timeout
        
        self.__initialize_model(model, pretrained)
        self.__initialize_priority_queue()    
        self.__create_output_directory(output_folder)
        meta_data_files, start_file = self.__get_names_meta_data_files(pathToMeta)
        self.__collect_urls(meta_data_files, start_file)
        
        #now download everything
        self.__download_urls()

def main():
        l = Laion400mDataset(num_elements_per_caption=666667, batch_size_meta=256, num_workers=16, shard_size=10000, thread_count=10)

if __name__ == "__main__":
    main()    