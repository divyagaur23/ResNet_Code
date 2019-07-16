# new
import os.path as pt
from resNet import ResNet
import torch.distributions as dist
from resNet import resnet18
from sacred import Experiment
from modules.helpers import AccuracyMetric
from modules.helpers import CrossEntropyLoss
# from dataset_loader import imagenet_loader
from modules.imagenet import ARGS_VAL
from modules.imagenet import ARGS_TRAIN_NONOISE_NOBLUR as ARGS_TRAIN
from modules.imagenet import make_image_loader
from modules.trainer import Trainer
from crumpets.pytorch.policy import PolyPolicy
# new

# cuda start
from torch.backends import cudnn
# cuda end
import torch.optim as optim

# experiment part

ex = Experiment()


# experiment configuration
@ex.config
def config():
    # dataset directory path
    datadir = '/b_test/folz/ILSVRC12_new'
    # snapshot directory
    outdir = '.'
    # learning rate
    lr = 0.01
    num_epochs = 90
    batch_size = 256
    nworkers = 18


# main for the experiment
@ex.automain
def main(lr, num_epochs, datadir, batch_size, nworkers, outdir):
    # code for GPU support : Start
    cudnn.benchmark = True
    network = resnet18().cuda()
    # code for GPU support : End
    # path = "changed_parameters/"
    # epochList = ["epoch_15.pth","epoch_20.pth","epoch_30.pth","epoch_90.pth","epoch_99.pth"]
    # snapshotLoad = torch.load("changed_parameters/epoch_99.pth")
    # network.load_state_dict(snapshotLoad.get("model_state"))
    train_iter = make_image_loader(
        pt.join(datadir, 'train.msgpack'),
        batch_size, nworkers, *ARGS_VAL
    )

    val_iter = make_image_loader(
        pt.join(datadir, 'val.msgpack'),
        batch_size, nworkers, *ARGS_VAL
    )

    # code without GPU support
    # net = resnet18()

    loss = CrossEntropyLoss(output_key="net_out").cuda()
    val_loss = CrossEntropyLoss(output_key="net_out").cuda()
    optimizer = optim.SGD(network.parameters(), lr=lr, weight_decay=0.0004, momentum=0.9)
    policy = PolyPolicy(optimizer, num_epochs, power=1)

    # trainer.logger.info(run_id=_run._id)
    # # trainer.set_hook('train_begin', set_eval)
    # with train_iter, val_iter:
    #     trainer.train(num_epochs, start_epoch=start_epoch)
    trainer = Trainer(
        network,
        optimizer,
        loss,
        AccuracyMetric(),
        None,
        policy,
        train_iter,
        val_iter,
        outdir,
        val_loss
    )
    with train_iter, val_iter:
        trainer.train(num_epochs)



