import numpy as np
import scipy.misc as sp
from labels import id2label
import os 
join = os.path.join


def kitti_to_cityscapes_instaces(instance_img):
    kitti_semantic = instance_img // 256
    kitti_instance = instance_img % 256
    print(kitti_semantic.max())
    print(kitti_instance.max())

    instance_mask = (kitti_instance > 0)
    cs_instance = (kitti_semantic*1000 + kitti_instance)*instance_mask + kitti_semantic*(1-instance_mask) 
    return cs_instance


if __name__ == '__main__':
    instanceSizes = {
        "bicycle"    : [] ,
        "caravan"    : [] ,
        "motorcycle" : [] ,
        "rider"      : [] ,
        "bus"        : [] ,
        "train"      : [] ,
        "car"        : [] ,
        "person"     : [] ,
        "truck"      : [] ,
        "trailer"    : [] ,
    }

    instanceCounts = instanceSizes.copy()
    category_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(sorted(instanceSizes.keys()))}

    #for split in ['training', 'testing']:
    for split in ['training']:
        instance_dir = join('/media/SSD_1TB/Kitti/data_semantics', split, 'instance/')
        instance_file_list = [f for f in os.listdir(instance_dir) if os.path.isfile(join(instance_dir, f))]

        for f in instance_file_list[:]:
            instance_img = sp.imread(join(instance_dir, f))
            instclassid_list = np.unique(instance_img)
            for instclassid in instclassid_list:
                instid = instclassid % 256 
                if instid > 0:
                    classid = instclassid // 256
                    mask = instance_img == instclassid 
                    instance_size = np.count_nonzero(mask)*1.0
                    instanceSizes[id2label[classid].name].append(instance_size)
                    instanceSizes[id2label[classid].name].append(1)

    print("Average instance sizes : ")
    for className in instanceSizes.keys():
        meanInstanceSize = np.nanmean(instanceSizes[className], dtype=np.float32)
        print('\"%s\"\t: %f,' % (className, meanInstanceSize))

    print("Average instance counts : ")
    for className in instanceSizes.keys():
        meanInstanceSize = len(instanceSizes[className])
        print('\"%s\"\t: %d,' % (className, meanInstanceSize))
    """
    Average instance sizes : 
    "bicycle"	: 912.851074,
    "caravan"	: 1263.285767,
    "motorcycle": 1091.125000,
    "rider"	    : 890.000000,
    "bus"	    : 3248.315674,
    "train"	    : 11109.500000,
    "car"	    : 3310.775879,
    "person"	: 872.750000,
    "truck"	    : 2062.683105,
    "trailer"	: 17606.199219,
    
    Average instance counts : 
    "bicycle"	: 94,
    "caravan"	: 14,
    "motorcycle": 16,
    "rider"	    : 58,
    "bus"	    : 38,
    "train"	    : 36,
    "car"	    : 3356,
    "person"	: 200,
    "truck"	    : 202,
    "trailer"	: 10,
    """


