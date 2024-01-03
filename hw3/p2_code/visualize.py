import matplotlib.pyplot as plt
import torch
import timm
import torch.nn.functional as F
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from model import *
from dataset import *
from utils import fixed_init

def draw_plot(attn_matx, id_list, img, tokenizer, output_path, n_cols=5):
    # check n_rows in subplots
    n_rows = len(id_list) // n_cols 
    if (len(id_list) // n_cols) != 0:
        n_rows += 1

    # Set up my subplots
    fig, ax = plt.subplots(n_rows, n_cols, figsize = (10, 6))
    for ax_row in ax:
        for ax_col in ax_row:
            ax_col.axis('off')

    attn_map_size = (24, 24)
    fig_size = (336, 336)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(fig_size), 
        transforms.ToPILImage()
    ])
    img = tfm(img) # PIL Image

    for i in range(len(id_list)):
        # [1, n_head, start_from_15: , 577]: 577 because the first one is start token for ViT
    
        attn_vector = attn_matx[:, :, -1*len(id_list) + i, 1:] 
        attn_vector = attn_vector.sum(dim = 1)

        attn_map = torch.reshape(attn_vector, attn_map_size)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max()) * 255
        attn_map = resize(attn_map.unsqueeze(0), fig_size)
        attn_map = attn_map.unsqueeze(0)
        # attn_map = F.interpolate(attn_map, size = fig_size, mode='bilinear')
        # print(torch.min(attn_map), torch.max(attn_map))
        fig.add_subplot(n_rows, n_cols, i+1)
        
        plt.imshow(img)
        plt.imshow(attn_map.squeeze(0).squeeze(0), alpha=0.6, cmap='jet')
        plt.title(tokenizer.decode([id_list[i]]))
        plt.axis('off')
        
    plt.savefig(output_path)

def visualize(model, model_path, encode_tfm, tokenizer, imgs_path, output_path, device):
    model = model.to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    
    img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path)]

    for name in tqdm(img_names): # set batch size = 1
        features = []
        temp = model.register_hook(features)

        img = Image.open(name).convert('RGB')
        img_encoded = encode_tfm(img)

        output_ids = model.greedy(img_encoded.unsqueeze(0).to(device))
        output_sentence = tokenizer.decode(output_ids)
        print(output_sentence)
        output_ids.insert(0, 50256)
        output_ids.append(50256)
        
        attn_matx = features[-1] # get the complete matrix for attention to each word
        
        draw_plot(attn_matx, output_ids, img, tokenizer, output_path=(output_path + '/' + name.split('/')[-1].split('.')[0]))
        
        for handle in temp:
            handle.remove()
        # print(output_sentence)
        # spreds_img_sentence[str(batch[1][0])] = output_sentence

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vits = [
        'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k', 
        'vit_large_patch14_clip_336.openai_ft_in12k_in1k'
    ]
    encoder = timm.create_model(vits[1], pretrained=True)
    encode_transform = create_transform(**resolve_data_config(encoder.pretrained_cfg, model=encoder))
    # print(encoder.feature_info)
    # print(encoder)

    cfg = Config(checkpoint='/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p2_data/decoder_model.bin')
    decoder = Decoder(cfg)

    tokenizer = BPETokenizer(
        encoder_file = '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/encoder.json',
        vocab_file = '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/vocab.bpe'
    )

    model = ImageCaptionTransformer(
        decoder = decoder, 
        encoder = encoder
    )

    tfm = transforms.Compose([
        transforms.ToTensor()
    ])

    val_set = ImageDataset(
        img_dir='/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p3_data/images', 
        tfm=encode_transform
    )

    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=True, num_workers=3, worker_init_fn=fixed_init(666)
    )

    output_path = './fig'
    model_path = '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/p2_code/ckpt/1best_PT_e5.ckpt'
    imgs_path = '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p3_data/images'
    visualize(model, model_path, encode_transform, tokenizer, imgs_path, output_path, device)