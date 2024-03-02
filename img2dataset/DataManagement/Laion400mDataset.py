from torch.utils.data import Dataset, DataLoader
import fsspec
import os
import torch
import open_clip
import pyarrow.parquet as pq
import heapq
import time
import tqdm

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
        #batch_idx = idx // self.batch_size
        #batch_start = batch_idx * self.batch_size
        #idx_within_batch = idx - batch_start

        #if batch_start != self.batch_start:
        #    self.batch_start = batch_start
        #    batch_end = min(batch_start + self.batch_size, self.total_rows)
        #    self.current_slice = self.dataframe.slice(batch_start, batch_end-batch_start).to_pandas()
        
        #caption = self.current_slice[idx_within_batch]
        #caption = self.current_slice.iloc[idx_within_batch]['caption']
        caption = self.current_slice.iloc[idx]['caption']
        if caption is None:
            return "", idx
        return caption, idx 


class Laion400mDataset(Dataset):
    def __initialize_model(self, model, pretrained):
        self.model, _, _ = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model)
        with torch.no_grad():
            self.caption_features_targets = self.model.encode_text(self.tokenizer(self.captions).to(self.device)) #this is a matrix of dimension Nx(dimension feature embedding)
            self.caption_features_targets /= self.caption_features_targets.norm(dim=-1, keepdim=True)

    def __initialize_priority_queue(self):
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
        if fs.isdir(url_path):
            input_files = sorted(fs.glob(url_path + "/*." + self.input_format))
            if len(input_files) == 0:
                raise ValueError(f"No file found at path {url_path} with extension {self.input_format}")
        else:
            input_files = [url_path]

        return input_files    

    def __create_output_directory(self, output_folder):
        self.output_folder = self.__make_path_absolute(output_folder)
        fs, output_path = fsspec.core.url_to_fs(self.output_folder)
        if not fs.exists(output_path):
            fs.mkdir(output_path)

    def __collect_urls(self, meta_data_files: list):
        for i, input_file in enumerate(meta_data_files):
            print("File_number " + str(i))
            self.__collect_urls_file(meta_data_file=input_file)

    def __collect_urls_file(self, meta_data_file):
        df = self.__download_meta_to_pyarrow_table(meta_data_file)
        df = self.__rename_cols_in_pyarrow_table(df)
        list_df = self.__divide_dataset_into_1m_shards(df)
        for i in range(len(list_df)):
            print("Shard number: " + str(i))
            dataset_url_cap = DatasetMetaData(list_df[i], self.tokenizer, self.batch_size_meta)
            def collate_fn(batch):
                with torch.no_grad():
                    caption, idx = zip(*batch)
                    return self.tokenizer(caption), idx
            dataloader = DataLoader(dataset_url_cap, batch_size=self.batch_size_meta, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers)#self.num_workers)
            self.__updatePriorityQueue(dataloader)

    def __divide_dataset_into_1m_shards(self, df):
        list_datasets = []
        len_dataset = len(df)
        len_shard = (len_dataset + 499999)//500000
        for i in range(len_shard):
            batch_start = i * 500000
            batch_end = min(batch_start + 500000, len_dataset)
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
                url_indices = batch[1] #welcher value in der priority
                for i in range(shortest_distance_np.shape[0]):
                    if(
                        (len(self.priority_queues[index_shortest_distance_np[i]]) < 
                        self.num_elements_per_caption) or 
                        (shortest_distance_np[i] > 
                        self.priority_queues[index_shortest_distance_np[i]][0][0])):

                        heapq.heappush(self.priority_queues[index_shortest_distance_np[i]], (shortest_distance_np[i], url_indices[i]))

                self.__cut_down_prriority_queues()
            stop = time.time()
            duration_batch = (stop-start)/self.batch_size_meta
            print("Duration")
            print(duration_batch)
    

    def __cut_down_prriority_queues(self):
        for i in range(len(self.priority_queues)):
            priority_queue_i = self.priority_queues[i]
            if(len(priority_queue_i) <= self.num_elements_per_caption):
                break
            for j in range(len(priority_queue_i)-self.num_elements_per_caption):
                heapq.heappop(priority_queue_i)                                 

    def __init__(self, num_elements_per_caption: int, batch_size_meta: int, 
                num_workers:int, output_folder: str = "laion400m-data", 
                pathToMeta: str = "laion400m-meta", process_count: int = 1, 
                thread_count: int = 1, image_size: int = 256, enable_wandb: bool = True,
                model: str = 'ViT-B-32', pretrained: str = 'laion400m_e32',
                captions: list = ["a dog", "a cat", "an airplane"]):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.columns_to_read = ["URL", "TEXT"]
        self.input_format = "parquet"
        self.output_format = "webdataset"
        self.captions = captions
        self.num_elements_per_caption = num_elements_per_caption
        self.batch_size_meta = batch_size_meta
        self.num_workers = num_workers
        self.k = 0
        
        self.__initialize_model(model, pretrained)
        self.__initialize_priority_queue()    
        self.__create_output_directory(output_folder)
        meta_data_files = self.__get_names_meta_data_files(pathToMeta)
        self.__collect_urls(meta_data_files)
        pass

def main():
    l = Laion400mDataset(num_elements_per_caption=666667, batch_size_meta=2048, num_workers=12)   


if __name__ == "__main__":
    main()    