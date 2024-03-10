from torch.utils.data import Dataset, DataLoader
import fsspec
import os
import torch
import open_clip
import pyarrow.parquet as pq
import pyarrow as pa
import heapq
import time
from datetime import timedelta
import pickle
import requests
import cv2
import numpy as np
import albumentations as A
from multiprocessing.pool import ThreadPool
from threading import Semaphore
import torch.nn as nn
import random


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

def make_path_absolute(path):
    fs, p = fsspec.core.url_to_fs(path)
    if fs.protocol == "file":
        return os.path.abspath(p)
    return path    

def download_resize_write(key_dist_url_caption, timeout, semaphore, output_dir_class):
    try:
        # Download image
        user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
        headers={"User-Agent": user_agent_string}
        print(key_dist_url_caption[1])
        response = requests.get(key_dist_url_caption[1], timeout=timeout, headers=headers)
        if response.status_code == 200:
            # Decode image
            img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
            # Resize image
            resized_img = resize_img(img)
            #write image to destination
            output_file_path = f'data/{output_dir_class}/{key_dist_url_caption[0]}.jpg'
            output_file_path = make_path_absolute(output_file_path).replace("\\", "/")
            cv2.imwrite(output_file_path, resized_img)
            semaphore.release()
            return None, None
        else:
            semaphore.release()
            print(f"Failed to download image from {key_dist_url_caption[1]}. Status code: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        semaphore.release()
        return None, None

class Merger():
    def __init__(self, prefix):
        self.prefix = prefix

    def __get_all_prqueue_paths(self, prefix):
        current_folder = make_path_absolute(".")
        fs_current, current_path = fsspec.core.url_to_fs(current_folder)
        if not fs_current.exists(current_path):
            fs_current.mkdir(current_path)
        files = fs_current.ls(current_path)
        list_prqueue_paths = []
        for file in files:
            name = file.split('/')[-1]
            if name.startswith("queue"):
                list_prqueue_paths.append(file)
        return list_prqueue_paths

    def __load_prqueues_from_path(self, list_prqueue_paths):
        list_prqueues = []
        for path in list_prqueue_paths:
            current_prqueue = pickle.load(path)
            list_prqueues.append(current_prqueue)
        return list_prqueues

    def __merge_priority_queues(self, list_prqueues, max_lenght_prqueue):
        final_prqueues=[]
        for i in range(len(list_prqueues[0])):
            current_class_queue = []
            for j in range(len(list_prqueues)):
                for tpl in list_prqueues[j][i]:
                    heapq.heappush(current_class_queue, tpl)
            self.__cut_down_priority_queue(current_class_queue, max_lenght_prqueue)
            
    def __cut_down_priority_queue(self, current_class_queue, max_lenght_prqueue):
        if(len(current_class_queue) <= max_lenght_prqueue):
            return
        for j in range(len(current_class_queue)-max_lenght_prqueue):
            heapq.heappop(current_class_queue)


    def merge(self, max_lenght_prqueue):
        list_prqueue_paths = self.__get_all_prqueue_paths(self.prefix)
        list_prqueues = self.__load_prqueues_from_path(list_prqueue_paths)
        final_prqueue = self.__merge_priority_queues(list_prqueues, max_lenght_prqueue)
        self.__save_prqueue(final_prqueue)


class DatasetMetaData(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.total_rows = len(dataframe)
        self.dataframe = self.dataframe.to_pandas()

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        caption = self.dataframe.iloc[idx]['caption']
        url = self.dataframe.iloc[idx]['url']
        if caption is None:
            return "", url
        return caption, url 

class Downloader():
    def __initialize_model(self, model):
        class TextEncoder(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, input):
                return self.model.encode_text(input)
        self.model, _, _ = open_clip.create_model_and_transforms(model, pretrained=self.pretrained)
        self.text_encoder = TextEncoder(self.model)          
        if torch.cuda.device_count() > 1:
            print(f"{torch.cuda.device_count()} GPUs are used")
            self.text_encoder = nn.DataParallel(self.text_encoder)
        self.text_encoder.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model)
        with torch.no_grad():
            n = len(self.templates)
            average = torch.zeros((len(self.captions), self.text_encoder(self.tokenizer("Hallo").to(self.device)).shape[1])).to(self.device)
            for i in range(n):
                list_full_captions = []
                for j in range(len(self.captions)):
                    list_full_captions.append(self.templates[i].format(self.captions[j]))

                list_full_captions_features = self.text_encoder(self.tokenizer(list_full_captions).to(self.device))
                list_full_captions_features /= list_full_captions_features.norm(dim=-1, keepdim=True)
                average += list_full_captions_features
            
        self.caption_features_targets = average/average.norm(dim=-1, keepdim=True)
        #self.caption_features_targets = self.text_encoder(self.tokenizer(self.captions).to(self.device)) #this is a matrix of dimension Nx(dimension feature embedding)
        #self.caption_features_targets /= self.caption_features_targets.norm(dim=-1, keepdim=True)

    def __initialize_priority_queue(self):
        prqueue = make_path_absolute(self.priority_queue_save_path)
        fs, prqueue_path = fsspec.core.url_to_fs(prqueue)
        file_path_queue = f"{prqueue_path}/queue.pkl"
        if fs.exists(file_path_queue):
            with fs.open(file_path_queue, 'rb') as f:
                self.priority_queues = pickle.load(f)
        else:
            self.priority_queues = []
            for i in range(len(self.captions)):
                self.priority_queues.append([])

    def __get_names_meta_data_files(self, pathToMeta, meta_from_to):
        pathToMeta = make_path_absolute(pathToMeta)
        fs, url_path = fsspec.core.url_to_fs(pathToMeta)
        self.fs = fs
        max = self.__get_max_input_file_filtered()
        if fs.isdir(url_path):
            input_files = sorted(fs.glob(url_path + "/*." + self.input_format))
            input_files = self.__filter_out_processed_non_relevant_files(input_files, max, meta_from_to)
        else:
            input_files = [url_path]

        start_file = max(max+1, meta_from_to[0])
        return input_files, start_file

    def __filter_out_processed_non_relevant_files(self, file_list, max, meta_from_to):
        filtered_file_list = []
        for path in file_list:
            name = path.split('/')[-1]
            number = int(name[5:10])
            if number > max and number >= meta_from_to[0] and number <= meta_from_to[1]:
                 filtered_file_list.append(path)
        return filtered_file_list

    def __get_max_input_file_filtered(self):
        prqueue = make_path_absolute(self.priority_queue_save_path)
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
        self.output_folder = make_path_absolute(output_folder)
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
        print("Number of shards: " + str(len(list_df)))        
        for i in range(len(list_df)):
            start = time.perf_counter()
            print("Shard number: " + str(i))
            dataset_url_cap = DatasetMetaData(list_df[i])
            dataloader = DataLoader(dataset_url_cap, batch_size=self.batch_size_meta, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers)#self.num_workers)
            self.__updatePriorityQueue(dataloader)
            elapsed_time = time.perf_counter()-start
            duration = timedelta(seconds=elapsed_time)
            duration_whole = timedelta(seconds=elapsed_time*len(list_df))
            print("Duration shard ", duration)
            print("Duration whole ", duration_whole)

            
    def __save_priority_queue(self, filenumber):
        self.__create_output_directory(self.priority_queue_save_path)
        prqueue = make_path_absolute(self.priority_queue_save_path)
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
            for batch in dataloader:
                caption_tokens = batch[0].to(self.device)
                caption_features = self.text_encoder(caption_tokens)
                caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)
                #calculate distances
                distances = caption_features @ self.caption_features_targets.T
                shortest_distance, index_shortest_distance = distances.max(dim=-1)
                shortest_distance_np = shortest_distance.to("cpu").numpy() # welche priorität
                index_shortest_distance_np = index_shortest_distance.to("cpu").numpy() #welche priority queue
                urls = batch[1] #welcher value in der priority
                for i in range(shortest_distance_np.shape[0]):
                    if(
                        (len(self.priority_queues[index_shortest_distance_np[i]]) < 
                        self.num_elements_per_caption) or 
                        (shortest_distance_np[i] < 
                        self.priority_queues[index_shortest_distance_np[i]][0][0])):

                        heapq.heappush(self.priority_queues[index_shortest_distance_np[i]], (shortest_distance_np[i], urls[i], self.tokenizer.decode(caption_tokens[i].cpu().numpy())))           
                self.__cut_down_prriority_queues()
    

    def __cut_down_prriority_queues(self):
        for i in range(len(self.priority_queues)):
            priority_queue_i = self.priority_queues[i]
            if(len(priority_queue_i) <= self.num_elements_per_caption):
                break
            for j in range(len(priority_queue_i)-self.num_elements_per_caption):
                heapq.heappop(priority_queue_i)


    def __download_urls(self):
        i = 0
        final_label_list = []
        for url_caption_list in self.priority_queues:
            url_caption_list = [(k, tuple_url_caption[1], tuple_url_caption[2]) for k, tuple_url_caption in enumerate(url_caption_list)]
            output_dir_class = make_path_absolute(f"data/{self.captions[i]}")
            final_label_list.extend(self.__create_label_list(url_caption_list, i))
            self.__create_output_directory(output_dir_class)
            self.__download_class(url_caption_list, self.captions[i])
            i += 1
        final_label_list_save = make_path_absolute('data')
        #save priority queue
        fs, final_label_list_save_path = fsspec.core.url_to_fs(final_label_list_save)
        file_path_queue = f"{final_label_list_save_path}/final_label_list.pkl"
        with fs.open(file_path_queue, 'wb') as f:
            pickle.dump(final_label_list, f)    
        

    def __create_label_list(self, key_url_caption_list, i):
        label_list = []
        for key_url_caption in key_url_caption_list:
            label_list.append((f"data/{self.captions[i]}/{key_url_caption[0]}.jpg", key_url_caption_list[2]))
        return label_list

    def __download_class(self, url_caption_list, output_dir_class):   
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

        with ThreadPool(self.thread_count) as thread_pool:
            for img_str, caption in thread_pool.imap_unordered(lambda dist_key_url_caption: download_resize_write(dist_key_url_caption, self.timeout, semaphore=semaphore, output_dir_class=output_dir_class),loader):
                pass

            thread_pool.terminate()
            thread_pool.join()
            del thread_pool                                                 

    def __init__(self, num_elements_per_caption: int, batch_size_meta: int, 
                num_workers:int, output_folder: str = "laion400m-data", 
                pathToMeta: str = "laion400m-meta",
                thread_count: int = 1, image_size: int = 256, timeout: int = 10,
                model: str = 'ViT-B-32', pretrained: str = './DownloadedModels/Model-B-32_Data-400M_Samples-34B_lr-5e-4_bs-32k.pt',
                meta_from_to: tuple=(1,31),
                captions: list = ["bird", "car", "chair", "dog", #VLCS: bird, car, chair, dog, person
                "elephant", "giraffe", "guitar", "horse", "house", "person"], #PACS: dog, elephant, giraffe, guitar, horse, house, person
                templates: list = ["a photo of a {}", "a picture of a {}", "a photo of my {}", "I love my {}", "This is a {}"],
                shard_size: int = 500000, priority_queue_save_path: str = "prqueue", number_save_files: str = 3):

        random.seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        self.pretrained = make_path_absolute(pretrained)
        self.templates = templates
        self.number_save_files = number_save_files
        print("batch_size: " + str(batch_size_meta))
        print("shard size " + str(shard_size))
        print("num_workers " + str(num_workers))
        print("meta_from_to " + str(meta_from_to))
        
        self.__initialize_model(model)
        self.__initialize_priority_queue()    
        self.__create_output_directory(output_folder)
        meta_data_files, start_file = self.__get_names_meta_data_files(pathToMeta, meta_from_to)
        print("meta_files " + str(meta_data_files))
        self.__collect_urls(meta_data_files, start_file)
        
        #now download everything
        #self.__download_urls()

def main():
        l = Downloader(num_elements_per_caption=200000, priority_queue_save_path="prqueue1", meta_from_to=(1,4), batch_size_meta=2048, num_workers=2, shard_size=200000, thread_count=48)
        #l = Downloader(num_elements_per_caption=666667, batch_size_meta=2, num_workers=0, shard_size=8, thread_count=10)

if __name__ == "__main__":
    main()    