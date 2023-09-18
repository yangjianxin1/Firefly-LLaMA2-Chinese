from os.path import join
import os
from loguru import logger
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, file):
        self.file = file

    def __iter__(self):
        with open(self.file, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                except:
                    break
                yield data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        return data


class PretrainDataProcessor(object):
    """
    数据预处理器，用于预处理数据，返回dataset。所有数据预处理器的父类。
    """
    def __init__(self, data_path, tokenizer, max_seq_length, min_seq_length, window_step_size, eval_size):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length    # 小于min_seq_length的序列，会被抛弃
        self.window_step_size = window_step_size    # 滑动窗口大小
        self.data_path = data_path
        self.tokenize_batch = 1024
        self.eval_size = eval_size

    def load_texts_from_file(self, file):
        """
        从文件中取出训练文本
        """
        if file.endswith('.jsonl'):
            df = pd.read_json(file, lines=True)
            text_list = [x.strip() for x in df['text'].tolist()]
        elif file.endswith('.csv'):
            df = pd.read_csv(file, sep='\t')
            text_list = [x.strip() for x in df['text'].tolist()]
        elif file.endswith('.txt'):
            with open(file) as f:
                text_list = [f.read().strip()]
        return text_list

    def slice_window_truncate(self, input_ids):
        """
        对input_ids，按照窗口大小，进行滑动截断。返回所有截断窗口。
        """
        windows = []
        for i in range(0, len(input_ids), self.window_step_size):
            window = input_ids[i: i+self.max_seq_length]
            # 小于min_seq_length的序列，则将其抛弃。
            if len(window) < self.min_seq_length and i > 0:
                continue
            windows.append(window)
        return windows

    def save_to_disk(self, obj, file):
        """
        将对象序列化的磁盘
        """
        with open(file, 'wb') as f:
            pickle.dump(obj, f)

    def load_from_disk(self, file):
        with open(file, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def load_dataset(self):
        """
        获取训练集和验证集
        """
        logger.info('Loading data from: {}'.format(self.data_path))
        # 创建缓存路径
        # todo 保存到output目录
        cache_dir = join(self.data_path, 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        # 读取缓存
        cache_file = join(cache_dir, 'train.pkl')
        if os.path.exists(cache_file):
            data_list = self.load_from_disk(cache_file)
            dataset = Dataset(data_list)
        else:
            # 收集所有训练文件路径
            files = []
            for root, dir_names, file_names in os.walk(self.data_path):
                for file_name in file_names:
                    file = join(root, file_name)
                    if file_name.endswith('.jsonl') or file_name.endswith('.csv') or file_name.endswith('.txt'):
                        files.append(file)
            logger.info(f'Total num of training file: {len(files)}')

            # 加载所有训练文本
            train_texts = []
            for file in tqdm(files):
                text_list = self.load_texts_from_file(file)
                train_texts += text_list
            logger.info(f'Total num of training text: {len(train_texts)}')

            # 对文本进行tokenize，并且使用窗口滑动进行截断
            logger.info(f'Start tokenizing data ...')
            train_windows = []  # 窗口截断之后的input_ids
            for i in tqdm(range(0, len(train_texts), self.tokenize_batch)):
                text_list = train_texts[i: i + self.tokenize_batch]
                input_ids = self.tokenizer(text_list).input_ids
                # 使用滑动窗口进行窗口截断
                for x in input_ids:
                    windows = self.slice_window_truncate(x)
                    train_windows += windows
            logger.info(f'Total training data num: {len(train_windows)}')
            # 缓存dataset对象到磁盘
            logger.info('Saving cache to disk ...')
            self.save_to_disk(train_windows, cache_file)

            dataset = Dataset(train_windows)

        # 将数据集，切分成训练集和验证集
        logger.info('Spliting train and eval dataset ...')
        if self.eval_size == 0:
            train_dataset = dataset
            eval_dataset = None
            logger.info(f'Num of train data: {len(train_dataset)}')
            logger.info(f'Num of eval data: 0')
        else:
            train_dataset, eval_dataset = train_test_split(dataset, test_size=self.eval_size)
            logger.info(f'Num of train data: {len(train_dataset)}')
            logger.info(f'Num of eval data: {len(eval_dataset)}')

        # 计算数据集的token数量
        total_token_num = 0
        for x in tqdm(train_dataset):
            total_token_num += len(x)
        logger.info(f'Total training token num: {total_token_num}')
        return train_dataset, eval_dataset

