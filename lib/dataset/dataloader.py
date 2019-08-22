import torch
import torch.utils.data as data

__all__ = ['Collate', 'MakeDataLoader']


class Collate(object):
    def __call__(self, batch):
        images, grids, boxes = [], [], []
        for obj in batch:
            if isinstance(obj, tuple):
                image, (grid, box) = obj
                images.append(image.unsqueeze(0))
                grids.append(grid)
                boxes.append(box)
            else:
                images.append(obj.unsqueeze(0))
        if len(grids) == 0:
            return torch.cat(images, dim=0)
        return torch.cat(images, dim=0), (grids, boxes)


def MakeDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=Collate(), pin_memory=False):
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )
