from PIL import Image
import torch

class Kmeans(torch.nn.Module):
    def __init__(self, img, k):
        super(Kmeans, self).__init__()
        self.m, self.n, self.channel = img.size()
        self.img = img.view(-1, self.channel)
        self.k = k
        self.train_on_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.train_on_gpu else 'cpu')
        self.means = torch.rand(self.k, self.channel).to(self.device)
        self.r = torch.full((self.img.size()[0],), self.k + 1, dtype=torch.long).to(self.device)
        
    def fit(self, max_iter = 300):
        for i in range(max_iter):
            dist = torch.sum((self.img[:, None] - self.means) ** 2, axis=2)
            new_r = torch.argmin(dist, axis=1)
            if torch.equal(self.r, new_r):
                break
            else:
                self.r = new_r
            for j in range(self.k):
                data_k = self.img[torch.nonzero(torch.where(self.r == j,torch.full_like(self.r, 1),torch.full_like(self.r, 0)), as_tuple=True)]
                if len(data_k) == 0:
                    self.means[j] = torch.rand(self.channel)
                else:
                    self.means[j] = torch.mean(data_k, axis=0)
        
    def predict(self):
        new_data = torch.round(self.means[self.r]*255)
        if self.train_on_gpu:
            disp = Image.fromarray(new_data.view(self.m, self.n, self.channel).cpu().numpy().astype('uint8'))
        else:
            disp = Image.fromarray(new_data.view(self.m, self.n, self.channel).numpy().astype('uint8'))
        disp.show(title='k-means')
        disp.save('k-means(pytorch)_'+str(self.k)+'.png')