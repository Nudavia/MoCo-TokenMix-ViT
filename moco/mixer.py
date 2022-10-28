# -*-coding:utf-8-*-

import numpy as np
import torch
from einops import repeat, rearrange
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

from utils import setup_seed, plot_img


def show_images(images, title=None):
    plot_img(images.cpu().numpy(), size=images.shape[0], title=title)


def show_patchs(patches, title=None):
    P = rearrange(patches, 'b t (c p1 p2) -> b t c p1 p2', p1=patch_size, p2=patch_size)
    plot_img(rearrange(P, 'b t c p1 p2 -> (b t) c p1 p2').cpu().numpy(), size=patches.shape[:2], title=title)


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 1, repeat(indexes, 'b t -> b t c', c=sequences.shape[-1]))


class PatchShuffle(torch.nn.Module):
    def __init__(self, group_size=32) -> None:
        super().__init__()
        self.group_size = group_size

    def forward(self, patches: torch.Tensor):
        B, T, C = patches.shape
        assert B % self.group_size == 0
        group_n = B // self.group_size
        # 每组的索引
        indexes = [random_indexes(T) for _ in range(group_n)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=0), dtype=torch.long).to(
            patches.device)
        forward_indexes = repeat(forward_indexes, 'n t -> (n g) t', g=self.group_size)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=0), dtype=torch.long).to(
            patches.device)
        backward_indexes = repeat(backward_indexes, 'n t -> (n g) t', g=self.group_size)

        patches = take_indexes(patches, forward_indexes)  # 将patch按index排序
        return patches, forward_indexes, backward_indexes


class PatchMix(torch.nn.Module):
    def __init__(self, mix_num=4, group_size=32) -> None:
        super().__init__()
        assert mix_num <= group_size
        self.mix_num = mix_num
        self.group_size = group_size

    def forward(self, patches: torch.Tensor):
        B, T, C = patches.shape
        assert T % self.mix_num == 0 and B % self.group_size == 0
        N = B // self.group_size  # 一个batch包含的组数
        S = T // self.mix_num  # 每个样本的patch分成的块数
        patches = rearrange(patches, '(n g) (m s) c ->n (g m) s c', n=N, s=S)

        L = patches.shape[1]
        for j in range(N):
            indexes = torch.zeros(L, dtype=torch.long).cuda()
            for i in range(L):
                indexes[i] = (i + i % self.mix_num * self.mix_num) % L
            patches[j] = torch.gather(patches[j], 0, repeat(indexes, 'l -> l s c', c=patches.shape[-1], s=S))

        target = torch.zeros((B, self.mix_num), dtype=torch.long).cuda()
        for i in range(B):
            target[i, :] = i // self.group_size * self.group_size + (
                    i % self.group_size + torch.arange(self.mix_num)) % self.group_size
        patches = rearrange(patches, 'n (g m) s c -> (n g) (m s) c', g=self.group_size)
        return patches, target


class Mix:
    def __init__(self, patch_size, group_size=32, mix_num=4, mix_p=1.0):
        self.patch_size = patch_size
        self.mix_p = mix_p
        self.shuffle = PatchShuffle(group_size=group_size)
        self.mix = PatchMix(mix_num=mix_num, group_size=group_size)

    @torch.no_grad()
    def __call__(self, X):
        N = X.shape[0]
        use_mix = np.random.rand() < self.mix_p
        if use_mix:
            patches = rearrange(X, 'b c (w p1) (h p2) -> b (w h) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
            patches, forward_indexes, backward_indexes = self.shuffle(patches)
            patches, target = self.mix(patches)
            patches = take_indexes(patches, backward_indexes)
            X = rearrange(patches, 'b (w h) (c p1 p2) -> b c (w p1) (h p2)', p1=self.patch_size, p2=self.patch_size,
                          w=int(np.sqrt(patches.shape[1])))
        else:
            # 只是在该GPU上的标签
            target = torch.arange(N, dtype=torch.long).cuda()
        return X, target, use_mix


if __name__ == '__main__':
    setup_seed()
    patch_size = 8
    batch_size = 16
    input_size = 32
    group_size = 8
    mix_num = 4
    dataset = datasets.CIFAR10('G:\CV-Datasets\official', train=True,
                               transform=Compose([ToTensor(), Resize(input_size), Normalize(0.5, 0.5)]))
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    classes = list(dataset.class_to_idx.keys())

    X, y = next(iter(dataloader))
    X = X.cuda()
    show_images(X, title='original')
    # mixer = Mix(patch_size, group_size, mix_num)
    # X, _, _ = mixer(X)
    # show_images(X, title='mix original')

    patches = rearrange(X, 'b c (w p1) (h p2) -> b (w h) (c p1 p2)', p1=patch_size, p2=patch_size)
    # 为了按原图位置展示
    patches = rearrange(patches, '(b1 b2) (w h) c -> (b1 w) (b2 h) c', b1=int(np.sqrt(patches.shape[0])),
                        w=int(np.sqrt(patches.shape[1])))
    show_patchs(patches, title='original patches')

    patches = rearrange(X, 'b c (w p1) (h p2) -> b (w h) (c p1 p2)', p1=patch_size, p2=patch_size)
    show_patchs(patches, 'patches')

    shuffle = PatchShuffle(group_size=group_size)
    mix = PatchMix(mix_num=mix_num, group_size=group_size)

    patches, forward_indexes, backward_indexes = shuffle(patches)
    show_patchs(patches, 'shuffle patches')
    patches, _= mix(patches)
    show_patchs(patches, 'mix patches')
    patches = take_indexes(patches, backward_indexes)
    show_patchs(patches, 'unshuffle patches')
    patches = rearrange(patches, '(b1 b2) (w h) c -> (b1 w) (b2 h) c', b1=int(np.sqrt(patches.shape[0])),
                        w=int(np.sqrt(patches.shape[1])))
    show_patchs(patches, 'original mix patches')
