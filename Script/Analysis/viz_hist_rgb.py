import os 
import sys
import torch 
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import random 
import torchvision.transforms as transforms
from PIL import Image
from data.rgbuc import RGBuvHistBlock, RGBuvHistBlock_Original
from torchvision.utils import save_image 
from tools.utils import normalize, save_pickle, load_pickle, make_folder, gif_generator
from plot import *

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA
import numpy as np 
from Script.Analysis.plot import unsupervised_scatter_plt, scatter_plt
from sklearn.cluster import FeatureAgglomeration, Birch, OPTICS, DBSCAN
import copy 
from einops import rearrange, repeat
import cv2
from matplotlib import pyplot as plt 
loader = transforms.Compose([transforms.ToTensor()]) 
histogram_block = RGBuvHistBlock(insz=224, h=32,  intensity_scale=False,  method='inverse-quadratic', device='cpu')
from math import log 
from collections import defaultdict 

make_folder(f"Samples/")

def plot_samples(Y, selected_name, DEST="Samples"):
    for cl in set(Y):
        make_folder(f"{DEST}/{cl}")
        try:
            os.system(f"rm {DEST}/{cl}/*")
        except:
            _ = 0 
        imgs  = list(selected_name[Y == cl])
        imgs =  random.sample(imgs , k= min(10, len(imgs)) )
        [os.system(f'cp {e} ./{DEST}/{cl}/')  for e in imgs]
        
def plot_gifs_samples(Y, selected_name, save_img=False, separated_selected = None):
    if separated_selected:
        for cl in separated_selected:
            make_folder(f"Samples/{cl}")
            # os.system(f"rm Samples/{cl}/*")
            imgs  = list(selected_name[Y == cl])
            images = [Image.open(e).resize( (384,192)).convert('RGB') for e in  imgs]
            if save_img:
                [e.save(f'Samples/{cl}/{i}.png') for i,e in enumerate(images)]
            gif_generator(images, name=f'Samples/{cl}/vid.gif')
    else:
        for cl in set(Y):
            make_folder(f"Samples/{cl}")
            # os.system(f"rm Samples/{cl}/*")
            imgs  = list(selected_name[Y == cl])
            images = [Image.open(e).resize( (384,192)).convert('RGB') for e in  imgs]
            if save_img:
                [e.save(f'Samples/{cl}/{i}.png') for i,e in enumerate(images)]
            gif_generator(images, name=f'Samples/{cl}/vid.gif')
        
def plot_gifs_images(Y, selected_name):
    for cl in set(Y):
        make_folder(f"Samples/{cl}")
        os.system(f"rm Samples/{cl}/*")
        imgs  = list(selected_name[Y == cl])
        [os.system(f"cp {e} Samples/{cl}/") for e in  imgs]

        
# 24 (Concat, Norm, 1000 , 32, sigma=0.001)
wt = 1000
dim=32
histogram_block = RGBuvHistBlock(insz=224, h=dim,  intensity_scale=False,  method='inverse-quadratic', device='cpu', sigma=0.001)






## OUTDATED    
vis_ccvid=False 
vis_custom=False  
Dump_names = False
cluster_feat = False 
gen_hist = False   
cluster_hist = False 
vis_hist = False 

## WORKING
dump_feats=False 
dump_ltcc=False 

vis_hist2 = False   # True 
vis_ltcc_hist2 = True    # True 

dump_celeb_feats=False  
vis_celeb_hist2=False

dump_concat_feats=False  
vis_concat_feats=False

vis_rgb_histogram=False
vis_rgb_uv_histogram_hyperparam= False
## OUTDATED
if vis_ccvid:
    ccvid='/data/priyank/synthetic/CCVID/CCVID/'
    selected = []
    for session in ['session1', 'session2', 'session3']:
        session_path = os.path.join(ccvid, session)
        for people in os.listdir(session_path):
            people_path = os.path.join(session_path, people)
            images = os.listdir(people_path)
            N = len(images)
            imgs =  random.sample(images, k=int(N * 0.1))
            selected += [os.path.join(people_path, e) for e in imgs]


    selected =  random.sample(selected, k=int(len(selected) * 0.1))
    for x in selected:
        image = Image.open(x)
        identifier = "_".join(x.split("/")[-3:])
        image.save(f"Samples/{identifier.replace('.jpg', '_rgb.png')}")
        image = loader(image)
        hist_image = histogram_block(image.unsqueeze(0))
        save_image(hist_image,  f"Samples/{identifier.replace('.jpg', '_hist.png')}")
        save_image(normalize(hist_image),  f"Samples/{identifier.replace('.jpg', '_hist_norm.png')}")

## OUTDATED        
if vis_custom:
    selected = ['t2.png', 't1.png']
    for x in selected:
        image = Image.open(x)
        image = loader(image)
        hist_image = histogram_block(image.unsqueeze(0))
        save_image(hist_image,  f"{x.replace('.png', '_hist.png')}")
        save_image(normalize(hist_image),  f"{x.replace('.png', '_normalize_hist.png')}")

## OUTDATED
if vis_hist:
    make_folder(f"Samples/")
    load_pickle_file = True 
    if load_pickle_file:
        selected = load_pickle("ccvid_hist")
        selected_name = load_pickle('ccvid_names')
        # torch.Size([347833, 12288])
        selected_sampled = selected[::20]
        selected_name = selected_name[::20]
        del selected
        
        # selected_sampled = rearrange(selected_sampled, "B (T H W) -> B T H W", T=3, H=32, W=32) * 100
        # save_image(selected_sampled[::100], "temp.png")

        # selected_sampled = selected_sampled.mean(1)
        # selected_sampled = rearrange(selected_sampled, "B H W -> B (H W)")
        selected_sampled = np.array(selected_sampled)
        selected_name = np.array(selected_name)
    
    print("Unsupervised - 1")
    ccvid_unsupervised = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled)
    unsupervised_scatter_plt(torch.tensor(ccvid_unsupervised) , name=f"ccvid_cl_hist_1", size_preference=25)

    print("Unsupervised Norm - 2")
    total_sum =  selected_sampled[0].sum(-1)
    selected_sampled_norm = selected_sampled / total_sum
    selected_sampled_norm = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled_norm)
    unsupervised_scatter_plt(torch.tensor(selected_sampled_norm) , name=f"ccvid_cl_hist_2", size_preference=25)

    print("Unsupervised L2 Norm - 3")
    selected_sampled_norm = torch.nn.functional.normalize(torch.tensor(selected_sampled).float(), p=0.5, dim=-1)
    selected_sampled_norm = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled_norm)
    unsupervised_scatter_plt(torch.tensor(selected_sampled_norm) , name=f"ccvid_cl_hist_3", size_preference=25)

    print("DBSCAN  - 4")
    dbscan = DBSCAN()
    Y = dbscan.fit_predict(selected_sampled)
    ccvid_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled)
    scatter_plt(X=torch.tensor(ccvid_embedded), Y=torch.tensor(Y), name=f"ccvid_cl_hist_4")
    
    print("DBSCAN L2 Norm - 5")
    selected_sampled_norm = torch.nn.functional.normalize(torch.tensor(selected_sampled).float(), p=0.5, dim=-1)
    dbscan = DBSCAN()
    Y = dbscan.fit_predict(selected_sampled_norm)
    ccvid_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled_norm)
    scatter_plt(X=torch.tensor(ccvid_embedded), Y=torch.tensor(Y), name=f"ccvid_cl_hist_5")
    # plot_samples(Y, selected_name)
    
    print("PCA - 6")
    pca = PCA(n_components=2).fit(selected_sampled_norm)
    pca_2d = pca.transform(selected_sampled_norm)
    unsupervised_scatter_plt(torch.tensor(pca_2d) , name=f"ccvid_cl_hist_6", size_preference=25)

    print("K Means - 7")
    ccvid_embedded = KMeans(n_clusters=45, random_state=0, n_init="auto").fit(selected_sampled_norm)
    Y = ccvid_embedded.predict(selected_sampled_norm)
    ccvid_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled_norm)
    scatter_plt(X=torch.tensor(ccvid_embedded), Y=torch.tensor(Y), name=f"ccvid_cl_hist_7")
    # plot_samples(Y, selected_name)

## OUTDATED
if cluster_feat:
    make_folder(f"Samples/")
    load_pickle_file = True 
    if load_pickle_file:
        selected = load_pickle("ccvid_hist")
        selected_name = load_pickle('ccvid_names')
        # torch.Size([347833, 12288])
        selected_sampled = selected[::20]
        selected_name = selected_name[::20]
        del selected
        
        # selected_sampled = rearrange(selected_sampled, "B (T H W) -> B T H W", T=3, H=64, W=64)
        # selected_sampled = selected_sampled.mean(1)
        # selected_sampled = rearrange(selected_sampled, "B H W -> B (H W)")
        selected_sampled = np.array(selected_sampled)  * 100
        selected_name = np.array(selected_name) * 100
    
    # dbscan = DBSCAN()
    # dbscan.fit(selected_sampled)
    # pca = PCA(n_components=2).fit(selected_sampled)
    # pca_2d = pca.transform(selected_sampled)
    # dbscan.labels_

    ccvid_embedded = KMeans(n_clusters=45, random_state=0, n_init="auto").fit(selected_sampled)
    # Y = ccvid_embedded.labels_
    # ccvid_embedded.cluster_centers_
    Y = ccvid_embedded.predict(selected_sampled)
    ccvid_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled)
    # pca = PCA(n_components=2).fit_transform(selected_sampled)
    scatter_plt(X=torch.tensor(ccvid_embedded), Y=torch.tensor(Y))
    for cl in set(Y):
        make_folder(f"Samples/{cl}")
        os.system(f"rm Samples/{cl}/*")
        imgs =  random.sample(list(selected_name[Y == cl]), k=10)
        [os.system(f'cp {e} ./Samples/{cl}/')  for e in imgs]
        
if gen_hist:
    os.system(f"rm -rf Samples")
    make_folder(f"Samples/")
    load_pickle_file = False 
    if load_pickle_file:
        selected = load_pickle("ccvid_hist2")
        selected_name = load_pickle('ccvid_names2')
        # torch.Size([347833, 12288])
        selected_sampled = selected[::20]
        selected_name = selected_name[::20]
        del selected
        selected_sampled = np.array(selected_sampled)
        selected_name = np.array(selected_name)
    else:
        ccvid='/data/priyank/synthetic/CCVID/CCVID/'
        selected = []
        names = [] 
        for session in ['session1', 'session2', 'session3']:
            session_path = os.path.join(ccvid, session)
            for people in os.listdir(session_path):
                people_path = os.path.join(session_path, people)
                images = os.listdir(people_path)
                for x in images:
                    name = os.path.join(people_path, x)
                    names.append( name )
                    img = Image.open( name )
                    hist = img.resize((224,224)).histogram()
                    hist = torch.tensor(hist)
                    selected.append( hist )
        selected = torch.stack(selected, 0)
        save_pickle(selected, "ccvid_hist2")
        save_pickle(names, "ccvid_names2")

## OUTDATED
if cluster_hist:
    make_folder(f"Samples/")
    load_pickle_file = True 
    if load_pickle_file:
        selected = load_pickle("ccvid_hist2")
        selected_name = load_pickle('ccvid_names2')
        # torch.Size([347833, 12288])
        selected_sampled = selected[::20]
        selected_name = selected_name[::20]
        del selected
        selected_sampled = np.array(selected_sampled)
        selected_name = np.array(selected_name)

    # ccvid_unsupervised = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled)
    # unsupervised_scatter_plt(torch.tensor(ccvid_unsupervised) , name=f"ccvid_cl_hist_1", size_preference=25)

    # total_sum =  selected_sampled[0].sum(-1)
    # selected_sampled = selected_sampled / total_sum
    # ccvid_unsupervised = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled)
    # unsupervised_scatter_plt(torch.tensor(ccvid_unsupervised) , name=f"ccvid_cl_hist_2", size_preference=25)

    selected_sampled = torch.nn.functional.normalize(torch.tensor(selected_sampled).float(), p=0.5, dim=-1)
    # ccvid_unsupervised_norm = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled)
    # unsupervised_scatter_plt(torch.tensor(ccvid_unsupervised_norm) , name=f"ccvid_cl_hist_3", size_preference=25)

    # ccvid_embedded = KMeans(n_clusters=45, random_state=0, n_init="auto").fit(selected_sampled)
    # Y = ccvid_embedded.predict(selected_sampled)
    # ccvid_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled)
    # scatter_plt(X=torch.tensor(ccvid_embedded), Y=torch.tensor(Y), name=f"ccvid_cl_hist_4")
    
    dbscan = DBSCAN()
    Y = dbscan.fit_predict(selected_sampled)
    ccvid_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled)
    scatter_plt(X=torch.tensor(ccvid_embedded), Y=torch.tensor(Y), name=f"ccvid_cl_hist_4")
    
    
    # pca = PCA(n_components=2).fit(selected_sampled)
    # pca_2d = pca.transform(selected_sampled)
    # dbscan.labels_
    # pca = PCA(n_components=2).fit_transform(selected_sampled)
    





## WORKING
if dump_feats:
    ccvid='/data/priyank/synthetic/CCVID/CCVID/'
    selected = []
    names = [] 
    for session in ['session1', 'session2', 'session3']:
        session_path = os.path.join(ccvid, session)
        for people in os.listdir(session_path):
            people_path = os.path.join(session_path, people)
            images = os.listdir(people_path)
            for x in images:
                name = os.path.join(people_path, x)
                names.append( name )
                image = loader(Image.open( name ).resize( (224,224)) ) 
                hist_image = histogram_block(image.unsqueeze(0))
                selected.append( hist_image.reshape(-1) )
    selected = torch.stack(selected)
    save_pickle(selected, "ccvid_hist")
    save_pickle(names, "ccvid_names")

if dump_ltcc:
    ltcc='/data/priyank/synthetic/LTCC/LTCC_ReID/train/'
    selected = []
    names = [] 
    images = os.listdir(ltcc)
    for x in images:
        name = os.path.join(ltcc, x)
        names.append( name )
        image = loader(Image.open( name ).resize( (224,224)) ) 
        hist_image = histogram_block(image.unsqueeze(0))
        selected.append( hist_image.reshape(-1) )
    selected = torch.stack(selected)
    save_pickle(selected, "ltcc_hist")
    save_pickle(names, "ltcc_names")


## WORKING
if dump_celeb_feats:
    
    celeb='/data/priyank/synthetic/DeepChange/DeepChangeDataset/train-set/'
    NAME="DeepChange2"

    celeb='/data/priyank/synthetic/Celeb-reID/train/'
    NAME="celeb_hist4"
    
    # histblock = RGBuvHistBlock(insz=224, h=32,  intensity_scale=False,  method='inverse-quadratic', device='cpu')
    histblock = RGBuvHistBlock(insz=224, h=32,  intensity_scale=False,  method='inverse-quadratic', device='cuda', sigma=0.001)
    selected = []
    names = [] 
    images = os.listdir(celeb)
    # images = images[::7]
    print(len(images))
    for x in images:
        # hist_image = np.zeros((256, 256, 256))
        name = os.path.join(celeb, x)
        names.append( name )

        
        img = Image.open( name ).resize( (224,224) )
        # img.save("rgb.jpg")

        # img = Image.open( name ).resize( (32,32) )
        # img.save("rgb.jpg")
        img = np.array(img)
        
        C = img.shape[2]
        H = img.shape[0]
        W = img.shape[1]
        HW = ( H * W )
        HWC = HW * C

        hist_image = cv2.calcHist([img],[0,1,2],None,[16,16,16],[0,256,0,256,0,256])
        hist_image = torch.tensor(hist_image).unsqueeze(0)
        # hist = cv2.calcHist([img],[0],None,[256],[0,256])
        # IDF = (hist != 0) * 1
        # for c in range(1, C):
        #     hist = cv2.calcHist([img],[c],None,[256],[0,256])
        #     IDF += (hist != 0) * 1 
        
        # hist_vector = []
        # for c in range(C):    
        #     local_vect = []
        #     hist = cv2.calcHist([img],[c],None,[256],[0,256])
        #     TF = hist / HW
        #     WT = TF 
        #     # WT = TF * (-np.log( (IDF +1e-6) / C )).clip(min=10 ** -6)
        #     # WT = (WT - WT.min()) / (WT.max() - WT.min()) 
        #     for i in range(H):
        #         for j in range(W):
        #             local_vect.append( WT[img[i,j,c]]  )
        #     hist_vector.append(torch.tensor(local_vect))
        # hist_image = torch.stack(hist_vector,0).unsqueeze(0)
        
        # image = loader( img ).cuda()
        # hist_image = histblock(image.unsqueeze(0))
        
        # save_image(normalize(hist_image), "hist.png")
        # hist_image = hist_image.mean(1) ; hist_image = rearrange(hist_image, "B H W -> B (H W)") 
        # hist_image = image.unsqueeze(0)
        # hist_image = rearrange(hist_image, "C H W -> 1 (C H W)") 

        hist_image = rearrange(hist_image, "1 C H W -> 1 (C H W)") 

        # hist_image = torch.nn.functional.normalize(hist_image.float(), p=2, dim=-1)
        selected.append(hist_image)
    
    selected = torch.stack(selected).cpu()
    # imgs = rearrange(selected, "B 1 (C H W) -> B C H W", C=3, H=32, W=32) 
    # save_image( imgs[:100] * 1000 , "hist.png")
    save_pickle(dict(name=names, vec=selected) , NAME)
    # quit()

## WORKING
if dump_concat_feats:
    load_pickle_file = False 
    ccvid='/data/priyank/synthetic/CCVID/CCVID/'
    selected = []
    names = [] 
    for session in ['session1', 'session2', 'session3']:
        session_path = os.path.join(ccvid, session)
        for people in os.listdir(session_path):
            people_path = os.path.join(session_path, people)
            images = os.listdir(people_path)
            for x in images[::2]:
                name = os.path.join(people_path, x)
                names.append( name )
                image = loader(Image.open( name ).resize( (224,224)) ) 
                hist_image = histogram_block(image.unsqueeze(0))
                selected.append( hist_image.reshape(-1) )
    selected = torch.stack(selected)
    save_pickle(selected, "ccvid_hist")
    save_pickle(names, "ccvid_names")



def scatter_rgb_all(R,G,B):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    s = 30
    ax.scatter(R, G, B, s=30, alpha=.6, edgecolor='k', lw=0.3)
    # ax.set_xlim3d(0, 255)
    # ax.set_ylim3d(0, 255)
    # ax.set_zlim3d(0, 255)
    ax.set_xlabel('Red', fontsize=14)
    ax.set_ylabel('Green', fontsize=14)
    ax.set_zlabel('Blue', fontsize=14)
    plt.savefig('rgb-scatter.png', bbox_inches='tight')

        

## WORKING
if vis_hist2:
    os.system(f"rm -rf Samples/*")
    make_folder(f"Samples/")
    load_pickle_file = True 
    if load_pickle_file:
        selected = load_pickle("ccvid_hist")
        selected_name = load_pickle('ccvid_names')
        # selected_sampled = selected[::20]
        # selected_name = selected_name[::20]
        selected_sampled = selected
        selected_name = selected_name
        del selected
        
        selected_sampled_display = rearrange(selected_sampled, "B (T H W) -> B T H W", T=3, H=32, W=32)
        # save_image(selected_sampled_display[::100]  * 100, "temp.png")
        
        selected_sampled = selected_sampled_display.mean(1) 
        selected_sampled = rearrange(selected_sampled, "B H W -> B (H W)") 
        selected_sampled = torch.nn.functional.normalize(selected_sampled.float(), p=2, dim=-1)

        print(selected_sampled.max(), selected_sampled.min())
        selected_sampled = np.array(selected_sampled)
        selected_name = np.array(selected_name)
    print(f" ... {len(selected_sampled)} .... ")
    print("TSNE FIRST")
    selected_sampled_norm = selected_sampled
    # selected_sampled_norm = torch.nn.functional.normalize(torch.tensor(selected_sampled).float(), p=0.5, dim=-1)
    # l2_norm = copy.deepcopy(selected_sampled_norm)
    # selected_sampled_norm = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(selected_sampled_norm)
    selected_sampled_norm = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(selected_sampled_norm)
    print("DBSCAN SECOND")
    # dbscan = DBSCAN()
    dbscan = DBSCAN(leaf_size=50, min_samples=10)
    Y = dbscan.fit_predict(selected_sampled_norm)
    
    ccvid_embedded = selected_sampled_norm[Y != -1]
    selected_name = selected_name[Y != -1]
    Y = Y[Y != -1]
    
    scatter_plt(X=torch.tensor(ccvid_embedded), Y=torch.tensor(Y), name=f"ccvid_cl_hist_8")
    # plot_gifs_samples(Y, selected_name)
    plot_gifs_images(Y, selected_name)
        
## WORKING
if vis_celeb_hist2:
    # https://web.engr.oregonstate.edu/~sheablyc/assets/docs/colors_of_love.pdf
    PICKLE_NAME="DeepChange2"
    PICKLE_NAME="celeb_hist4"
    NAME = "Samples_32/"
    os.system(f"rm -rf {NAME}/*")
    make_folder(f"{NAME}/")
    
    
    selected = load_pickle(PICKLE_NAME)
    H,W = 224, 224
    H,W = 32, 32
    selected_name = selected["name"]
    vec = selected["vec"].cpu()
    selected_name = np.array(selected_name)

    imgs = rearrange(vec, "B 1 C -> B C")

    if False:
        # img = rearrange(imgs, "B (C H W) -> B C H W", H=224, W=224, C=3) 
        img = rearrange(imgs, "B (C H W) -> B C H W", H=H, W=W, C=3) 
        save_image(normalize(img[:10]) * 10000 , "hist.png")
    
    # imgs = rearrange(imgs, "B (C D) -> B C D", C=3, D=H * W).sum(1)
    # imgs = rearrange(imgs, "B (C D) -> B C D", C=3, D=H * W).mean(1)
    # imgs = imgs.clip(min=10 ** -6)    
    # imgs.mean(-1)
    # imgs = imgs * 10000 
    print(imgs.shape)
    
    vec = torch.nn.functional.normalize(imgs.float(), p=2, dim=-1) 
    selected_sampled = np.array(vec.squeeze(1))
    
    
    
    print("TSNE FIRST")
    selected_sampled_norm = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=10).fit_transform(selected_sampled)
    # unsupervised_scatter_plt(X=torch.tensor(selected_sampled_norm), PADDING = 10, name="temp", size_preference=20)
    scatter_rgb_all(selected_sampled_norm[:,0],selected_sampled_norm[:,1],selected_sampled_norm[:,2])
    

    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=20, random_state=0, max_iter=1000).fit(selected_sampled)
    # Y = kmeans.predict(selected_sampled)
    # scatter_plt(X=torch.tensor(selected_sampled_norm), Y=torch.tensor(Y), name=f"temp2")
    # plot_samples(Y, selected_name)
    # plot_samples(Y, selected_name, DEST=NAME)

    print(" ***  DBSCAN ***  ")
    dbscan = DBSCAN(min_samples=1)
    DBSCAN_X = dbscan.fit_predict(selected_sampled_norm)
    ids = set(DBSCAN_X)
    ids= [id for id in ids if len(DBSCAN_X[DBSCAN_X == id]) != 1 ]

    indices = DBSCAN_X == ids[0]
    for id in ids: indices = (indices) | (DBSCAN_X == id)
    plot_samples(DBSCAN_X[indices], selected_name[indices], DEST=NAME)
    
    
    # plot_samples(DBSCAN_X, selected_name, DEST=NAME)

    # celeb_embedded = selected_sampled_norm[DBSCAN_X != -1]
    # selected_name = selected_name[DBSCAN_X != -1]
    # # img = img[DBSCAN_X != -1]
    # DBSCAN_X = DBSCAN_X[DBSCAN_X != -1]
    
    # for i in range(10):save_image( normalize( img[DBSCAN_X == i] ) * 10000 , f"hist_{i}.png") 
    

    # print(" ***  DBSCAN ***  ")
    # dbscan = DBSCAN()
    # DBSCAN_X = dbscan.fit_predict(selected_sampled)
    # print(set(DBSCAN_X))


## WORKING
if vis_ltcc_hist2:
    os.system(f"rm -rf Samples/*")
    make_folder(f"Samples/")
    
    selected = load_pickle("ltcc_hist")
    selected_name = load_pickle('ltcc_names')
    
    selected_sampled = selected
    # norm = False 
    norm = True 
    if norm:
        selected_sampled_display = rearrange(selected_sampled, "B (T H W) -> B T H W", T=3, H=32, W=32)
        # save_image(selected_sampled_display[::100]  * 100, "temp.png")
        selected_sampled = selected_sampled_display.mean(1) 
        selected_sampled = rearrange(selected_sampled, "B H W -> B (H W)") 
        selected_sampled = torch.nn.functional.normalize(selected_sampled.float(), p=2, dim=-1)


    print(selected_sampled.max(), selected_sampled.min())
    selected_sampled = np.array(selected_sampled)
    selected_name = np.array(selected_name)

    print(f" ... {len(selected_sampled)} .... ")
    print("TSNE FIRST")
    selected_sampled_norm = selected_sampled
    
    selected_sampled_norm = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=2).fit_transform(selected_sampled_norm)
    # selected_sampled_norm = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(selected_sampled_norm)
    
    print("DBSCAN SECOND")
    # dbscan = DBSCAN(leaf_size=50, min_samples=4)
    dbscan = DBSCAN(leaf_size=50, min_samples=1)
    Y = dbscan.fit_predict(selected_sampled_norm)
    print(set(Y))
    ccvid_embedded = selected_sampled_norm[ (Y != -1) ]
    selected_name = selected_name[ (Y != -1) ]
    Y = Y[(Y != -1)]
    counts = []
    neglect_element = [1,2,3]
    selected_pids = []
    for e in set(Y):
        count=  (Y == e).sum()
        if count not in neglect_element:
            selected_pids.append(e)
            counts.append(  count )


    print(selected_pids)
    # print(counts)
    # scatter_plt(X=torch.tensor(ccvid_embedded), Y=torch.tensor(Y), name=f"ltcc_cl_hist_8")
    plot_gifs_samples(Y, selected_name, save_img=False , separated_selected=selected_pids )
    # plot_gifs_images(Y, selected_name)
    
    
    
        


if vis_concat_feats:
    ccvid='/data/priyank/synthetic/CCVID/CCVID/'

    os.system(f"rm -rf Samples/*")
    make_folder(f"Samples/")
    load_pickle_file = True 
    
    selected = load_pickle("ccvid_hist")
    selected_name = load_pickle('ccvid_names')
    selected_sampled = selected[::2]
    selected_name = selected_name[::2]
    del selected
    
    selected_sampled = normalize(selected_sampled)
    selected_sampled = selected_sampled.float() * wt

    N = range(len(selected_name))
    indices = random.sample(N, k=10)
    [os.system(f"cp {selected_name[e]} Samples/") for e in indices]
    name_dict = {e: selected_name[e].split("/")[-1] for e in indices}
    name_hist_dict = {e: selected_name[e].split("/")[-1].replace(".", "_hist.") for e in indices}

    selected_sampled_display = rearrange(selected_sampled, "B (T H W) -> B T H W", T=3, H=dim, W=dim)
    [save_image(selected_sampled_display[e], f"Samples/{name_hist_dict[e]}") for e in indices]

    selected_sampled = selected_sampled_display.mean(1) 
    selected_sampled = rearrange(selected_sampled, "B H W -> B (H W)") 
    selected_sampled = torch.nn.functional.normalize(selected_sampled.float(), p=2, dim=-1)

    selected_sampled = np.array(selected_sampled)
    selected_name = np.array(selected_name)
    print(f" ... {len(selected_sampled)} .... ")
    print("TSNE FIRST")
    selected_sampled_norm = selected_sampled
    
    selected_sampled_norm = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(selected_sampled_norm)
    print("DBSCAN SECOND")
    
    # dbscan = DBSCAN(leaf_size=50, min_samples=10)
    dbscan = DBSCAN()
    Y = dbscan.fit_predict(selected_sampled_norm)
    
    ccvid_embedded = selected_sampled_norm[Y != -1]
    selected_name = selected_name[Y != -1]
    Y = Y[Y != -1]
    
    # scatter_plt(X=torch.tensor(ccvid_embedded), Y=torch.tensor(Y), name=f"ccvid_cl_hist_8")
    # plot_gifs_samples(Y, selected_name)
    plot_gifs_images(Y, selected_name)


if vis_rgb_histogram:
    selected_imgs = ['00033.jpg', '00064.jpg', '00099.jpg',  '00111.jpg',
        '00115.jpg', '00140.jpg', '00154.jpg', '00210.jpg', '00218.jpg', '00306.jpg']
    done = []
    ccvid='/data/priyank/synthetic/CCVID/CCVID/'
    for session in ['session1', 'session2', 'session3']:
        session_path = os.path.join(ccvid, session)
        for people in os.listdir(session_path):
            people_path = os.path.join(session_path, people)
            images = os.listdir(people_path)
            for x in images:
                if (x not in selected_imgs) or (x in done):
                    continue 
                done.append(x)
                name = os.path.join(people_path, x)
                name_to_be_saved = name.split("/")[-1][:-4]
                image = loader(Image.open( name ).resize( (224,224)) ) 
                hist_image = cv2.calcHist([ image.permute(1,2,0).numpy() ],[0,1,2],None,[20,20,20],[0,1,0,1,0,1])

                hist_image = hist_image.astype("float")
                hist_image /= hist_image.sum()
                
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(111, projection="3d")
                ALPHA=0.4
                # ax.w_xaxis.pane.set_color('w')
                ax.w_xaxis.pane.set_alpha(ALPHA)
                # ax.w_yaxis.pane.set_color('w')
                ax.w_yaxis.pane.set_alpha(ALPHA)
                # ax.w_zaxis.pane.set_color('w')
                ax.w_zaxis.pane.set_alpha(ALPHA)
                

                plt.setp( ax.get_xticklabels(), visible=False)
                plt.setp( ax.get_yticklabels(), visible=False)
                plt.setp( ax.get_zticklabels(), visible=False)
                # plt.axis('off')
                # ax.tick_params(axis='x', which='major', pad=10, bottom=False, top=False,labelbottom=False)
                # ax.tick_params(axis='y', which='major', pad=10, bottom=False, top=False,labelbottom=False)
                # ax.tick_params(axis='z', which='major', pad=10, bottom=False, top=False,labelbottom=False)

                # For each bin in the histogram
                cmap = plt.get_cmap('viridis')
                max_x , max_y, max_z = 0,0,0
                for (x, y, z), val in np.ndenumerate(hist_image):
                    # If the bin is not empty
                    if val > 0:
                        # Create a cuboid with size proportional to the bin value
                        # ax.scatter(x, y, z, s=val*5000, facecolors=plt.cm.viridis(val))
                        # ax.scatter(x, y, z, s=val*5000, facecolors=cmap(val))
                        ax.scatter(x, y, z, s=val*30000, facecolors="#f8a600", zorder=12, lw=0.25, ec="black", alpha=0.9)
                        max_x = max(x, max_x)
                        max_y = max(x, max_y)
                        max_z = max(x, max_z)


                ax.grid(axis = "y", color="#A8BAC4", lw=1.2)
                ax.grid(axis = "z", color="#A8BAC4", lw=1.2)
                ax.grid(axis = "x", color="#A8BAC4", lw=1.2)

                ax.xaxis.set_tick_params(length=6, width=1.2)

                ax.xaxis.set_ticks([i for i in range(0, max_x+5, 5)])
                ax.yaxis.set_ticks([i for i in range(0, max_y+5, 5)])
                ax.zaxis.set_ticks([i for i in range(0, max_z+5, 5)])

                # # Make gridlines be below most artists.
                ax.set_axisbelow(True)

                # # # Add grid lines
                ax.grid(axis = "both", color="#A8BAC4", lw=.5)

                # Remove all spines but the one in the bottom
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)

                # Customize bottom spine
                ax.spines["bottom"].set_lw(1.2)
                ax.spines["bottom"].set_capstyle("butt")
                
                
                # Set labels and title
                ax.set_xlabel("Red", fontsize=40).set_rotation(-20)
                ax.set_ylabel("Green", fontsize=40).set_rotation(90)
                ax.set_zlabel("Blue", fontsize=40).set_rotation(270)
                # ax.set_title("3D Color Histogram")
                plt.savefig(f"{name_to_be_saved}-RGB Hist.png", pad_inches=0.0)
                
                        

if vis_rgb_uv_histogram_hyperparam:
    selected_imgs = ['00033.jpg', '00064.jpg', '00099.jpg',  '00111.jpg',
        '00115.jpg', '00140.jpg', '00154.jpg', '00210.jpg', '00218.jpg', '00306.jpg']
    done = []
    ccvid='/data/priyank/synthetic/CCVID/CCVID/'
    session = 'session1' 
    people = '111_06'
    x =  '00045.jpg'
    session_path = os.path.join(ccvid, session)
    people_path = os.path.join(session_path, people)
    name = os.path.join(people_path, x)

    name_to_be_saved = name.split("/")[-1][:-4]
    image = loader(Image.open( name ).resize( (224,224)) ) 

    save_image(image, f"Samples/RGB.png")

    his_bin_size = [20, 32, 48, 64]
    for hist in his_bin_size:
        histogram_block = RGBuvHistBlock(insz=224, h=hist,  intensity_scale=False,  method='inverse-quadratic', device='cpu')
        hist_image = histogram_block(image.unsqueeze(0))
        hist_image = normalize(hist_image) 
        hist_image = 1000 * hist_image
        save_image(hist_image, f"Samples/{hist}.png")



    for weight_factor in [1, 10, 100, 1000]:
        histogram_block = RGBuvHistBlock(insz=224, h=64,  intensity_scale=False,  method='inverse-quadratic', device='cpu')
        hist_image = histogram_block(image.unsqueeze(0))
        hist_image = normalize(hist_image) 
        hist_image = weight_factor * hist_image
        save_image(hist_image, f"Samples/64_{weight_factor}.png")


    for sigma in [0.001, 0.01, 0.02, 0.5, 1]:
        histogram_block = RGBuvHistBlock(insz=224, h=64,  intensity_scale=False,  method='inverse-quadratic', device='cpu', sigma=sigma)
        hist_image = histogram_block(image.unsqueeze(0))
        hist_image = normalize(hist_image) 
        hist_image = 1000 * hist_image
        save_image(hist_image, f"Samples/64_1000_{sigma}.png")


    import pdb
    pdb.set_trace()
    
        
    
    
    
    hist_image = cv2.calcHist([ image.permute(1,2,0).numpy() ],[0,1,2],None,[20,20,20],[0,1,0,1,0,1])

    hist_image = hist_image.astype("float")
    hist_image /= hist_image.sum()
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection="3d")
    ALPHA=0.4
    # ax.w_xaxis.pane.set_color('w')
    ax.w_xaxis.pane.set_alpha(ALPHA)
    # ax.w_yaxis.pane.set_color('w')
    ax.w_yaxis.pane.set_alpha(ALPHA)
    # ax.w_zaxis.pane.set_color('w')
    ax.w_zaxis.pane.set_alpha(ALPHA)
    

    plt.setp( ax.get_xticklabels(), visible=False)
    plt.setp( ax.get_yticklabels(), visible=False)
    plt.setp( ax.get_zticklabels(), visible=False)
    # plt.axis('off')
    # ax.tick_params(axis='x', which='major', pad=10, bottom=False, top=False,labelbottom=False)
    # ax.tick_params(axis='y', which='major', pad=10, bottom=False, top=False,labelbottom=False)
    # ax.tick_params(axis='z', which='major', pad=10, bottom=False, top=False,labelbottom=False)

    # For each bin in the histogram
    cmap = plt.get_cmap('viridis')
    max_x , max_y, max_z = 0,0,0
    for (x, y, z), val in np.ndenumerate(hist_image):
        # If the bin is not empty
        if val > 0:
            # Create a cuboid with size proportional to the bin value
            # ax.scatter(x, y, z, s=val*5000, facecolors=plt.cm.viridis(val))
            # ax.scatter(x, y, z, s=val*5000, facecolors=cmap(val))
            ax.scatter(x, y, z, s=val*30000, facecolors="#f8a600", zorder=12, lw=0.25, ec="black", alpha=0.9)
            max_x = max(x, max_x)
            max_y = max(x, max_y)
            max_z = max(x, max_z)


    ax.grid(axis = "y", color="#A8BAC4", lw=1.2)
    ax.grid(axis = "z", color="#A8BAC4", lw=1.2)
    ax.grid(axis = "x", color="#A8BAC4", lw=1.2)

    ax.xaxis.set_tick_params(length=6, width=1.2)

    ax.xaxis.set_ticks([i for i in range(0, max_x+5, 5)])
    ax.yaxis.set_ticks([i for i in range(0, max_y+5, 5)])
    ax.zaxis.set_ticks([i for i in range(0, max_z+5, 5)])

    # # Make gridlines be below most artists.
    ax.set_axisbelow(True)

    # # # Add grid lines
    ax.grid(axis = "both", color="#A8BAC4", lw=.5)

    # Remove all spines but the one in the bottom
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Customize bottom spine
    ax.spines["bottom"].set_lw(1.2)
    ax.spines["bottom"].set_capstyle("butt")
    
    
    # Set labels and title
    ax.set_xlabel("Red", fontsize=40).set_rotation(-20)
    ax.set_ylabel("Green", fontsize=40).set_rotation(90)
    ax.set_zlabel("Blue", fontsize=40).set_rotation(270)
    # ax.set_title("3D Color Histogram")
    plt.savefig(f"{name_to_be_saved}-RGB Hist.png", pad_inches=0.0)
    
    



# hist = cv2.normalize(hist_image, hist_image).flatten()
# # Create a 3D histogram
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Construct arrays for the anchor positions of the bars.
# xpos, ypos = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = 0

# # Construct arrays with the dimensions for the bars.
# dx = dy = np.ones_like(zpos)
# dz = hist

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

# plt.show()
                
               


# cd ~/MADE_ReID/
# python Script/Analysis/viz_hist_rgb.py