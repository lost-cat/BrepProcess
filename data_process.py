from dataProcessor import DeepCadDataProcessor

if __name__ == '__main__':
    processor = DeepCadDataProcessor(data_dir='/media/huangxinyu/disk/数据/data',
                                     save_dir='/media/huangxinyu/disk/data')
    processor.process()
