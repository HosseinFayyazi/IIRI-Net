from prettytable import PrettyTable
import pickle
import torch
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


model_path = 'IO/final/sinc/1_saved_model.pth'
with open(model_path, 'rb') as file:
    trainer = CPU_Unpickler(file).load()
trainer.gpu = False
model = {'CNN_model_par': trainer.CNN_net.state_dict(),
         'DNN1_model_par': trainer.DNN1_net.state_dict(),
         'DNN2_model_par': trainer.DNN2_net.state_dict()
         }
table = PrettyTable(["Modules", "Parameters"])
total_params = 0
print('CNN-Net')
for name, parameter in trainer.CNN_net.named_parameters():
    if not parameter.requires_grad: continue
    params = parameter.numel()
    table.add_row([name, params])
    total_params += params
print(table)
print(f"Total Trainable Params of CNN_net: {total_params}")

table = PrettyTable(["Modules", "Parameters"])
total_params = 0
print('CNN-Net')
for name, parameter in trainer.DNN1_net.named_parameters():
    if not parameter.requires_grad: continue
    params = parameter.numel()
    table.add_row([name, params])
    total_params += params
print(table)
print(f"Total Trainable Params of DNN1_net: {total_params}")

table = PrettyTable(["Modules", "Parameters"])
total_params = 0
print('CNN-Net')
for name, parameter in trainer.DNN2_net.named_parameters():
    if not parameter.requires_grad: continue
    params = parameter.numel()
    table.add_row([name, params])
    total_params += params
print(table)
print(f"Total Trainable Params of DNN2_net: {total_params}")
