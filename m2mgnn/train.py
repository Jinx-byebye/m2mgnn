import torch
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid, LINKXDataset, HeterophilousGraphDataset
import argparse
from torch import nn
import numpy as np
import uuid
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
from model import M2MGNN


parser = argparse.ArgumentParser()
parser.add_argument('--patience', type=int, default=100,
                    help='Early stopping.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--wd1', type=float, default=0.01,
                    help='Weight decay.')
parser.add_argument('--wd2', type=float, default=0.01,
                    help='Weight decay.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden unit.')
parser.add_argument('--nlayers', type=int, default=64,
                    help='Number of convolutional layers.')
parser.add_argument('--dataset', type=str, default='texas',
                    help='Data set to be used.')
parser.add_argument('--c', type=int, default=5,
                    help='Number of edge types.')
parser.add_argument('--lamda', type=float, default=0.0,
                    help='Strength of reg on soft label diversity.')
parser.add_argument('--dropout2', type=float, default=0.7,
                    help='Dropout rate for raw feature.')
parser.add_argument('--beta', type=float, default=0.4,
                    help='Strength of ego-feature.')
parser.add_argument('--temperature', type=float, default=1.,
                    help='Temperature.')
parser.add_argument('--remove_self_loop', type=bool, default=True,
                    help='If set to True, dont use self loop.')
parser.add_argument('--device', type=str, default='0',
                    help='gpu id.')


args = parser.parse_args()
device = 'cuda:'+ args.device if torch.cuda.is_available() else 'cpu'


citation = ['texas', 'wisconsin', 'cornell']
if args.dataset in citation:
    dataset = WebKB(root='data/', name=args.dataset)
elif args.dataset in ['squirrel', 'chameleon']:
    dataset = WikipediaNetwork(root='data/', name=args.dataset)
elif args.dataset == 'actor':
    dataset = Actor(root='data/Actor')
elif args.dataset in ["penn94"]:
    dataset = LINKXDataset(root='data/', name=args.dataset)
elif args.dataset in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(root='data/', name=args.dataset, split='geom-gcn')
else:
    dataset = HeterophilousGraphDataset(root='data/', name=args.dataset)

data = dataset[0].to(device)
checkpt_file = 'trained_model_dict/' + uuid.uuid4().hex + '.pt'

data.edge_index, _ = add_remaining_self_loops(data.edge_index)

if args.dataset in ["penn94"]:
    num_split = 5
else:
    num_split = 10

def train_step(train_mask, model):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask]) + args.lamda * model.reg
    loss.backward()
    optimizer.step()
    return loss


def val_step(val_mask, model):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        loss = criterion(out[val_mask], data.y[val_mask])
        acc = int((pred[val_mask] == data.y[val_mask]).sum()) / int(val_mask.sum())
        return loss.item(), acc


def test_step(test_mask, model):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        loss = criterion(out[test_mask], data.y[test_mask])
        acc = int((pred[test_mask] == data.y[test_mask]).sum()) / int(test_mask.sum())
        return loss.item(), acc


# begin training
acc_list = []

for i in range(num_split):
    split = i % num_split
    train_mask = data.train_mask[:, split]
    val_mask = data.val_mask[:, split]
    test_mask = data.test_mask[:, split]
    #
    model = M2MGNN(in_feat=dataset.num_features, hid_feat=args.hidden, out_feat=dataset.num_classes,
                  num_layers=args.nlayers, dropout=args.dropout, c=args.c, beta=args.beta, dropout2=args.dropout2,
                  temperature=args.temperature, remove_self_loof=args.remove_self_loop).to(device)

    optimizer = torch.optim.Adam([
                        {'params':model.params1,'weight_decay':args.wd1},
                        {'params':model.params2,'weight_decay':args.wd2},
                        ], lr=args.lr)


    criterion = nn.CrossEntropyLoss()
    best = 100
    for j in range(1000):
        train_loss = train_step(train_mask, model)
        val_loss, val_acc = val_step(val_mask, model)
        if val_loss < best:
            count = 0
            best = val_loss
            torch.save(model.state_dict(), checkpt_file)

        else:
            count += 1
            if count == args.patience:
                break
        # print(f'epoch : {j}; train loss: {format(train_loss, ".3f")}; val loss: {format(val_loss, ".3f")}; '
        #       f'val acc: {format(val_acc, ".3f")}')
    model.load_state_dict(torch.load(checkpt_file))
    loss, best_acc = test_step(test_mask, model)


    print('---------------------------------------')
    print(f'intermediate result: {best_acc}')
    acc_list.append(best_acc)

print('------------------------------------------')
print(f'final acc list: {acc_list}')
print(f'{args.dataset}: average acc: {np.mean(acc_list)}; std : {np.std(acc_list)}')

