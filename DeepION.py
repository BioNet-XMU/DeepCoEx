import random
from byol_pytorch import BYOL
from torchvision import models
from sklearn.preprocessing import MinMaxScaler
from Augmentation import *
import umap
from neg_augment import neg_transform


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 示例：设置随机种子为42
set_seed(50)


def img2tensor(img):
    img = img.transpose((2, 3, 0, 1))
    img = img.astype(np.float32)
    img = torch.from_numpy(img).cuda()
    return img


def DeepION_training(input_filename, image_size, mode,tensor_path):
    print('Step 1: Start DeepION training ...')
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    torch.cuda.set_device(2)
    oridata = np.load(input_filename)   # data type is npy
    oridata = oridata.T
    target_height=image_size[0]
    oridata = oridata / np.sum(oridata, axis=0).reshape(1, -1)[0]
    data = MinMaxScaler().fit_transform(oridata)

    resnet = models.resnet18(pretrained=True).cuda()
    current_gpu = torch.cuda.current_device()
    print(f"Current GPU device index: {current_gpu}")

    # 负增强
    ## NOTING: 'argument_fn' is the negative augmentation function 
    argument_fn = torch.nn.Sequential(
        neg_transform(p=1,tensor_path=tensor_path,target_height=target_height),
    )

    argument_fn2 = torch.nn.Sequential(
        ColorJitter(0.8, 0.8, 0, p=1),
        RandomBoxBlur((5, 5), p=0.5),
        IntensityDependentMissing(p=1),
        RandomMissing(p=1)
    )

    argument_fn3 = torch.nn.Sequential(
        ColorJitter(0.8, 0.8, 0, p=1),
        RandomBoxBlur((5, 5), p=0.5),
        IntensityDependentMissing(p=1),
        RandomMissing(p=1)
    )

    learner = BYOL(resnet, image_size=image_size, hidden_layer='avgpool',
                   augment_fn=argument_fn, augment_fn2=argument_fn2, augment_fn3=argument_fn3)

    opt = torch.optim.Adam(learner.parameters(), lr=2e-4)

    mini_batch = 100
    num = len(data[0])

    log_path = "epoch_avg_loss.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch\tavg_loss\n")

    for epoch in range(50):
        index = np.arange(len(data[0]))
        np.random.shuffle(index)
        data = data[:, index]

        total_loss = 0.0
        step = 0  # 记录本 epoch 里算了几次 loss（batch 数）

        if num < 2:
            raise ValueError("At least two ion images are required.")

        batch_ranges = [
            (i, min(i + mini_batch, num))
            for i in range(0, num, mini_batch)
        ]

        # Avoid a final batch containing only one sample
        if len(batch_ranges) > 1:
            start, end = batch_ranges[-1]
            if end - start == 1:
                prev_start, prev_end = batch_ranges[-2]
                batch_ranges[-2] = (prev_start, prev_end - 1)
                batch_ranges[-1] = (prev_end - 1, end)

        for batch, (start, end) in enumerate(batch_ranges):
            current_batch_size = end - start

            image_array = data[:, start:end]
            image_array = image_array.reshape(
                image_size[0], image_size[1], current_batch_size, 1
            )
            image_array = np.concatenate(
                [image_array, image_array, image_array], axis=3
            )
            image_tensor = img2tensor(image_array)  

            loss = learner(image_tensor)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()

            loss_val = float(loss.item())
            total_loss += loss_val
            step += 1

            print(f'epoch {epoch}: batch {batch} loss: {loss.item():.4f}')    
        
        avg_loss = total_loss / max(step,1)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch}\t{avg_loss:.6f}\n")
            
        print(f"[epoch {epoch}] avg_loss = {avg_loss:.6f}  (saved to {log_path})")

    torch.save(resnet.state_dict(), mode + '_ResNet18_params.pth')


def DeepION_predicting(input_filename, image_size, mode):
    print('Step 2: Start DeepION predicting ...')

    oridata = np.load(input_filename)
    oridata = oridata.T
    oridata = oridata / np.sum(oridata, axis=0).reshape(1, -1)[0]
    data = MinMaxScaler().fit_transform(oridata)

    resnet = models.resnet18(pretrained=True).cuda()
    resnet.load_state_dict(torch.load(mode + '_ResNet18_params.pth'))
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

    mini_batch = 100
    num = len(data[0])
    features = np.zeros((num, 512))

    for batch in range(num // mini_batch):
        with torch.no_grad():
            image_array = data[:, batch * mini_batch: (batch + 1) * mini_batch]
            image_array = image_array.reshape(image_size[0], image_size[1], mini_batch, 1)
            image_array = np.concatenate([image_array, image_array, image_array], axis=3)
            image_tensor = img2tensor(image_array)
            embedding = resnet(image_tensor)
            embedding = embedding[:, :, 0, 0]
            embedding = embedding.detach().cpu().numpy()
            features[batch * mini_batch:(batch + 1) * mini_batch] = embedding

    with torch.no_grad():
        image_array = data[:, (batch + 1) * mini_batch:]
        image_array = image_array.reshape(image_size[0], image_size[1], len(image_array[0]), 1)
        image_array = np.concatenate([image_array, image_array, image_array], axis=3)
        image_tensor = img2tensor(image_array)

        feature = resnet(image_tensor)
        feature = feature[:, :, 0, 0]
        feature = feature.detach().cpu().numpy()
        features[(batch + 1) * mini_batch:] = feature

    return features


def DimensionalityReduction(features):
    print('Step 3: Start Dimensionality Reduction ...')
    return_feature = umap.UMAP(n_components=20, metric='cosine', random_state=0).fit_transform(features)
    return_feature = MinMaxScaler().fit_transform(return_feature)
    return return_feature