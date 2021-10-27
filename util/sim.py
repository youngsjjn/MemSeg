import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F

mem_root = 'exp/rugd/deeplab18mem_os16_mem24_3444_noLosses/model/memory_150.pt'
if mem_root:
    m_items = torch.load(mem_root, map_location=lambda storage, loc: storage.cuda())

trte = cosine_similarity(m_items.cpu().numpy(), dense_output=True)
# trte[trte>0.99]==0.
trte_min = np.min(trte)
trte_max = np.max(trte[trte<0.99])
trte_mean = np.mean(trte[trte<0.99])

if True:

    import matplotlib.pyplot as plt

    plt.imshow(trte)
    # plt.colorbar(orientation='vertical')
    plt.show()

    plt.imsave('mem_sim_loss000.png', trte)

print(trte)