import data.img_transforms as T
from data.dataloader import DataLoaderX
from data.dataset_loader import *
from data.samplers import *


from torch.utils.data import ConcatDataset, DataLoader

from data.datasets.ltcc import *
from data.datasets.prcc import *
from data.datasets.mevid import *
from data.datasets.ccvid import *


import data.spatial_transforms as ST
import data.temporal_transforms as TT

__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
    'mevid': MEVID,
    'ccvid': CCVID,     
}



def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(config.DATA.DATASET, __factory.keys()))
    kwargs = {}
    if config.DATA.RANDOM_FRAMES:
        kwargs["load_as_random_frames"] = True
    
    if config.DATA.DATASET_SAMPLING_PERCENTAGE:
        dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT,aux_info=config.DATA.AUX_INFO,meta_dir=config.DATA.META_DIR,meta_dims=config.MODEL.META_DIMS[0], sample_dataset=config.DATA.DATASET_SAMPLING_PERCENTAGE, test_mode=config.TEST.MODE, **kwargs)
    else:
        dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT,aux_info=config.DATA.AUX_INFO,meta_dir=config.DATA.META_DIR,meta_dims=config.MODEL.META_DIMS[0], test_mode=config.TEST.MODE, **kwargs)
    return dataset


def build_img_transforms(config):
    transform_train = T.Compose([
        T.Resize((config.DATA.IMG_HEIGHT , config.DATA.IMG_WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=config.AUG.RE_PROB)
    ])
    transform_test = T.Compose([
        T.Resize((config.DATA.IMG_HEIGHT , config.DATA.IMG_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test



def build_vid_transforms(config):
    spatial_transform_train = ST.Compose([
        ST.Scale((config.DATA.IMG_HEIGHT, config.DATA.IMG_WIDTH), interpolation=3),
        ST.RandomHorizontalFlip(),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ST.RandomErasing(height=config.DATA.IMG_HEIGHT, width=config.DATA.IMG_WIDTH, probability=config.AUG.RE_PROB)
    ])
    spatial_transform_test = ST.Compose([
        ST.Scale((config.DATA.IMG_HEIGHT, config.DATA.IMG_WIDTH), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
        temporal_transform_train = TT.TemporalDivisionCrop(size=config.AUG.SEQ_LEN)
    elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
        temporal_transform_train = TT.TemporalRandomCrop(size=config.AUG.SEQ_LEN, 
                                                         stride=config.AUG.SAMPLING_STRIDE)
    else:
        raise KeyError("Invalid temporal sempling mode '{}'".format(config.AUG.TEMPORAL_SAMPLING_MODE))

    temporal_transform_test = None

    return spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test


def build_dataloader(config, local_rank=None, teacher_mode=None, eval_mode=None):
    kwargs = {}
    dataset = build_dataset(config)
    # image dataset
    transform_train, transform_test = build_img_transforms(config)
    # transform_train = build_transform(config,is_train=True)
    # transform_test = build_transform(config,is_train=False)
    
    if config.DATA.SAMPLING_PERCENTAGE and config.DATA.SAMPLING_PERCENTAGE != 100:
        print( f"\n\n\n ONLY USING {config.DATA.SAMPLING_PERCENTAGE}% of Test Pids \n\n\n\n ")
        train_sampler = DistributedRandomIdentitySampler_Percent(dataset.train, num_instances=config.DATA.NUM_INSTANCES, seed=config.SOLVER.SEED, percent= config.DATA.SAMPLING_PERCENTAGE)
    else:
        train_sampler = DistributedRandomIdentitySampler(dataset.train,
            num_instances=config.DATA.NUM_INSTANCES, seed=config.SOLVER.SEED)
        
    kwargs['datatset_name'] = config.DATA.DATASET
    dataset_fix = config.DATA.DATASET_FIX
    if config.TRAIN.COLOR_PROFILE:
        kwargs['color_profile'] = config.TRAIN.COLOR_PROFILE

    IMAGE_DATASET= ImageDataset
    if dataset_fix:
        kwargs[dataset_fix] = True 
    if config.DATA.F8:
        kwargs['F8'] = True 
    if config.TRAIN.DEBUG:
        kwargs['Debug'] = True 
    if config.TEST.MODE:
        kwargs['return_index'] = True 
     
    if config.DATA.GREY_SCALE:
        kwargs['grey_scale'] = True 
    if config.MODEL.JEPA_ENABLE:
        kwargs['return_jepa_paths'] = True

    ### VIDEO  
    if config.TRAIN.TRAIN_VIDEO:
        del transform_train
        spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test = build_vid_transforms(config)
        VID_DATASET = VideoDataset    
        trainloader = DataLoaderX(
                dataset=VID_DATASET(dataset=dataset.train, spatial_transform=spatial_transform_train, temporal_transform=temporal_transform_train, train=True, **kwargs),
                sampler=train_sampler, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY, drop_last=True, local_rank=local_rank)

        # split each original test video into a series of clips and use the averaged feature of all clips as its representation
        queryloader = DataLoaderX(
            dataset=VID_DATASET(dataset=dataset.recombined_query, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test, train=False, **kwargs),
            sampler=DistributedInferenceSampler(dataset.recombined_query),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY, drop_last=False, shuffle=False, local_rank=local_rank)
        galleryloader = DataLoaderX(
            dataset=VID_DATASET(dataset=dataset.recombined_gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test, train=False, **kwargs),
            sampler=DistributedInferenceSampler(dataset.recombined_gallery),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY, drop_last=False, shuffle=False, local_rank=local_rank)

        return trainloader, queryloader, galleryloader, dataset, train_sampler, [queryloader, galleryloader]

    ### IMGAE 
    if 'mevid' in config.DATA.DATASET or 'ccvid' in config.DATA.DATASET:
        IMAGE_DATASET = Video_as_Image
        if dataset_fix:
            IMAGE_DATASET = Video_as_Image_fixes
        
        trainloader = DataLoaderX(dataset=IMAGE_DATASET(dataset=dataset.train, transform=transform_train,aux_info=config.DATA.AUX_INFO, train=True , **kwargs),
                                sampler=train_sampler,
                                batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=config.DATA.PIN_MEMORY, drop_last=True, local_rank=local_rank)

        # split each original test video into a series of clips and use the averaged feature of all clips as its representation
        queryloader = DataLoaderX(
            dataset=IMAGE_DATASET(dataset=dataset.recombined_query, transform=transform_test, aux_info=config.DATA.AUX_INFO, train=False, **kwargs),
            sampler=DistributedInferenceSampler(dataset.recombined_query),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY, drop_last=False, shuffle=False, local_rank=local_rank)
        galleryloader = DataLoaderX(
            dataset=IMAGE_DATASET(dataset=dataset.recombined_gallery, transform=transform_test, aux_info=config.DATA.AUX_INFO, train=False, **kwargs),
            sampler=DistributedInferenceSampler(dataset.recombined_gallery),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY, drop_last=False, shuffle=False, local_rank=local_rank)
        return trainloader, queryloader, galleryloader, dataset, train_sampler, [queryloader, galleryloader]
    else:
        if dataset_fix:
            IMAGE_DATASET = ImageDataset_fixes

        trainloader = DataLoaderX(dataset=IMAGE_DATASET(dataset=dataset.train, transform=transform_train,aux_info=config.DATA.AUX_INFO, train=True, **kwargs ),
                                sampler=train_sampler,
                                batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=config.DATA.PIN_MEMORY, drop_last=True, local_rank=local_rank)

        galleryloader = DataLoaderX(dataset=IMAGE_DATASET(dataset=dataset.gallery, transform=transform_test,aux_info=config.DATA.AUX_INFO, train=False, **kwargs),
                                sampler=DistributedInferenceSampler(dataset.gallery),
                                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=config.DATA.PIN_MEMORY, drop_last=False, shuffle=False, local_rank=local_rank)
        
        if 'prcc' in config.DATA.DATASET:
            queryloader_same = DataLoaderX(dataset=IMAGE_DATASET(dataset=dataset.query_same, transform=transform_test,aux_info=config.DATA.AUX_INFO, train=False, **kwargs),
                                    sampler=DistributedInferenceSampler(dataset.query_same),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=config.DATA.PIN_MEMORY,drop_last=False, shuffle=False, local_rank=local_rank)
            queryloader_diff = DataLoaderX(dataset=IMAGE_DATASET(dataset=dataset.query_diff, transform=transform_test,aux_info=config.DATA.AUX_INFO, train=False, **kwargs),
                                    sampler=DistributedInferenceSampler(dataset.query_diff),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False, local_rank=local_rank)


            combined_dataset = ConcatDataset([queryloader_diff.dataset, galleryloader.dataset])

            val_loader = DataLoader(
                dataset=combined_dataset,
                batch_size=config.DATA.TEST_BATCH,
                num_workers=config.DATA.NUM_WORKERS,
                pin_memory=config.DATA.PIN_MEMORY,
                drop_last=False,
                shuffle=False
            )

            combined_dataset_same = ConcatDataset([queryloader_same.dataset, galleryloader.dataset])

            val_loader_same = DataLoader(
                dataset=combined_dataset_same,
                batch_size=config.DATA.TEST_BATCH,
                num_workers=config.DATA.NUM_WORKERS,
                pin_memory=config.DATA.PIN_MEMORY,
                drop_last=False,
                shuffle=False
            )
            
            # import torch.distributed as dist
            # print(dist.get_rank(), len(dataset.gallery), len(galleryloader), len(DistributedInferenceSampler(dataset.gallery)), len(val_loader))
            # 0 3384 17 1692 70                                                                                                                                                                    
            # 1 3384 17 1692 70 
            # 0 3384 34 3384 70
            
            
            
            return trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same
        else:
            queryloader = DataLoaderX(dataset=IMAGE_DATASET(dataset=dataset.query, transform=transform_test, train=False, aux_info=config.DATA.AUX_INFO, **kwargs),
                                    sampler=DistributedInferenceSampler(dataset.query),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False, local_rank=local_rank)

            combined_dataset = ConcatDataset([queryloader.dataset, galleryloader.dataset])

            val_loader = DataLoader(
                dataset=combined_dataset,
                batch_size=config.DATA.TEST_BATCH,
                num_workers=config.DATA.NUM_WORKERS,
                pin_memory=config.DATA.PIN_MEMORY,
                drop_last=False,
                shuffle=False
            )

            # import torch.distributed as dist
            # print(dist.get_rank(), len(dataset.gallery), len(galleryloader), len(DistributedInferenceSampler(dataset.gallery)), len(val_loader))
            # 0 125353 1254 125353 1356
            
            # 0 125353 627 62677 1356
            # 1 125353 627 62677 1356
            
            return trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader

        

    
