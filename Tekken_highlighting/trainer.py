import time
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import vis_tool
from config import get_config

from models.model import CNN, GRU

class Trainer(object):
    def __init__(self, config, h_loader, r_loader, test_loader):
        self.config = config
        self.h_loader = h_loader
        self.r_loader = r_loader
        self.test_loader = test_loader

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay

        self.n_epochs = config.n_epochs
        self.n_steps = config.n_steps
        self.log_interval = int(config.log_interval)  # in case
        self.checkpoint_step = int(config.checkpoint_step)

        self.use_cuda = config.cuda
        self.outf = config.outf
        self.build_model()
        self.vis = vis_tool.Visualizer()

    def build_model(self):
        self.c2d = CNN().cuda()
        self.c2d.load_state_dict(torch.load('cnn.pkl'))  # load pre-trained cnn extractor

        for l,p in self.c2d.named_parameters():
            p.requires_grad = False

        self.gru = GRU(self.c2d).cuda()


    def train(self):
        # create optimizers
        cfig = get_config()
        opt = optim.RMSprop(filter(lambda p: p.requires_grad, self.gru.parameters()),
                         lr=self.lr,
                         weight_decay=self.weight_decay)

        start_time = time.time()
        criterion = nn.BCELoss()

        max_acc = 0.

        for epoch in range(self.n_epochs):
            self.gru.train()
            epoch_loss = []
            for step, (h, r) in enumerate(zip(self.h_loader, self.r_loader)):
                h_video = h
                r_video = r

                # highlight video
                h_video = Variable(h_video).cuda()
                r_video = Variable(r_video).cuda()

                self.gru.zero_grad()

                predicted = self.gru(h_video)

                target = torch.ones(predicted.shape, dtype=torch.float32).cuda()

                h_loss = criterion(predicted, target)  # compute loss

                h_loss.backward()
                opt.step()

                self.gru.zero_grad()

                predicted = self.gru(r_video)  # predicted snippet's score

                target = torch.zeros(predicted.shape, dtype=torch.float32).cuda()
                r_loss = criterion(predicted, target)  # compute loss

                r_loss.backward()
                opt.step()

                step_end_time = time.time()

                total_loss = r_loss + h_loss
                epoch_loss.append((total_loss.data).cpu().numpy())

                print('[%d/%d][%d/%d] - time: %.2f, h_loss: %.3f, r_loss: %.3f, total_loss: %.3f'
                      % (epoch + 1, self.n_epochs, step + 1, self.n_steps, step_end_time - start_time, h_loss, r_loss,
                         total_loss))

                self.vis.plot('H_LOSS with lr:%.4f, b1:%.1f, b2:%.3f, wd:%.5f'
                              % (cfig.lr, cfig.beta1, cfig.beta2, cfig.weight_decay),
                              (h_loss.data).cpu().numpy())

                self.vis.plot('R_LOSS with lr:%.4f, b1:%.1f, b2:%.3f, wd:%.5f'
                              % (cfig.lr, cfig.beta1, cfig.beta2, cfig.weight_decay),
                              (r_loss.data).cpu().numpy())

                # if step == 3: break
                # if step == 2: break

            self.vis.plot("Avg loss plot", np.mean(epoch_loss))



            if epoch % self.checkpoint_step == 0:
                accuracy, savelist = self.test(self.test_loader)

                if accuracy > max_acc:
                    max_acc = accuracy
                    torch.save(self.gru.state_dict(), './samples/lr_%.4f_chkpoint' % cfig.lr + str(epoch + 1) + '.pth')
                    for f in savelist:
                        np.save("./samples/" + f[0][0] + ".npy", f[1])
                    print(np.load("./samples/testRV04(198,360).mp4.npy"))
                    print("checkpoint saved")


    def test(self, t_loader):

        # Test accuracy
        self.gru.eval()
        test_avg_acc = 0.
        test_cnt = 0
        savelist = []

        for idx, (video, label, filename) in enumerate(self.test_loader):
            video = Variable(video).cuda()
            predicted = self.gru(video)  # [ frame 수, 1]

            predicted = predicted.view(1, -1)
            predicted = predicted.cpu().detach().numpy()

            predicted = predicted[0]
            label = label.cpu().numpy()

            # print(type(predicted), type(label))

            gt_label_predicted_score = predicted * label
            gt_label_predicted_score = list(gt_label_predicted_score)

            # gt_label_predicted_score = gt_label_predicted_score.cpu().numpy()
            # print("Highlight frame predicted score:", gt_label_predicted_score)

            # print(gt_label_predicted_score)
            # print(gt_label_predicted_score.shape)

            # print(gt_label_predicted_score)

            for sc in gt_label_predicted_score[0]:
                if sc != 0.:
                    print("%.3f" % sc, end=' ')

            for i in range(len(predicted)):
                if predicted[i] >= 0.45:
                    predicted[i] = 1.
                else:
                    predicted[i] = 0.

            # print("After threshold predicted:", predicted)
            # print("Actual label:", label)

            acc = (predicted == label).sum().item() / float(len(predicted))
            print("filename: %s accuracy: %.4f" % (filename, acc))
            test_avg_acc += acc
            test_cnt += 1

            savelist.append([filename, predicted])

            print()

        test_avg_acc = test_avg_acc / test_cnt

        print("Accuracy:", round(test_avg_acc, 4))
        self.vis.plot("Accuracy with lr:%.3f" % self.lr, test_avg_acc)

        return test_avg_acc, savelist