import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse, os, sys, random, logging, time, json

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Running custom PyTorch models in Edge Impulse')
"""
# Data / run args
parser.add_argument('--data-directory', type=str, default='data')
parser.add_argument('--out-directory', type=str, default='out')

# Baseline arguments to always show
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning-rate', type=float, default = 0.01)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--num_conv', type=int, default=3, choices=[1,2,3])
parser.add_argument('--num_linear', type=int, default=1, choices=[1,2])
parser.add_argument('--network_width', type=float, default=0.5, help="Width multiplier for channels")
parser.add_argument('--noise_std', type=float, default=0, help="Gaussian noise stddev during training")
parser.add_argument('--channel_growth_mode', type=int, default=3, choices=[0,1,2,3,4,5])
parser.add_argument('--dendritic-optimization', type=str, required=False, default="false")
#Only show if dendritic optimization is checked
parser.add_argument('--switch_speed', type=str, default='slow', help="speed to switch", choices=['slow', 'medium', 'fast'])
parser.add_argument('--max_dendrites', type=int, default=3)
parser.add_argument('--improvement_threshold', type=str, default='low', choices=['high', 'medium', 'low'])
parser.add_argument('--dendrite_weight_initialization_multiplier', type=float, default=0.01)
parser.add_argument('--dendrite_forward_function', type=str, default='tanh', choices=['relu','sigmoid','tanh'], help="0=sigmoid,1=relu,2=tanh")
parser.add_argument('--dendrite-conversion', type=str, default='All Layers', choices=['Linear Only','All Layers'])
parser.add_argument('--improved-dendritic-optimization', type=str, required=False, default="false")
#Only show if improved is checked
parser.add_argument('--perforated-ai-token', type=str, required=False, default="")
"""
# Data / run args
parser.add_argument('--data-directory', type=str, default='data')
parser.add_argument('--out-directory', type=str, default='out')

# Baseline arguments to always show
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning-rate', type=float, default = 0.001)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--num_conv', type=int, default=3, choices=[1,2,3])
parser.add_argument('--num_linear', type=int, default=2, choices=[1,2])
parser.add_argument('--network_width', type=float, default=2, help="Width multiplier for channels")
parser.add_argument('--noise_std', type=float, default=0, help="Gaussian noise stddev during training")
parser.add_argument('--channel_growth_mode', type=int, default=5, choices=[0,1,2,3,4,5])
parser.add_argument('--dendritic-optimization', type=str, required=False, default="false")

# Only show if dendritic optimization is checked
parser.add_argument('--switch_speed', type=str, default='slow', help="speed to switch", choices=['slow', 'medium', 'fast'])
parser.add_argument('--max_dendrites', type=int, default=3)
parser.add_argument('--improvement_threshold', type=str, default='medium', choices=['high', 'medium', 'low'])
parser.add_argument('--dendrite_weight_initialization_multiplier', type=float, default=0.01)
parser.add_argument('--dendrite_forward_function', type=str, default='tanh', choices=['relu','sigmoid','tanh'], help="0=sigmoid,1=relu,2=tanh")
parser.add_argument('--dendrite-conversion', type=str, default='All Layers', choices=['Linear Only','All Layers'])
parser.add_argument('--improved-dendritic-optimization', type=str, required=False, default="false")

# Only show if improved is checked
parser.add_argument('--perforated-ai-token', type=str, required=False, default="")

args, unknown = parser.parse_known_args()

os.environ["PAIEMAIL"] = "user@edgeimpulse.com"
os.environ["PAITOKEN"] = args.perforated_ai_token

if not os.path.exists(args.out_directory):
    os.makedirs(args.out_directory, exist_ok=True)

def str2bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "t", "yes", "y")

# grab train/test set
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

# Split test set in half for test and validation
split_idx = len(X_test) // 2
X_val = X_test[:split_idx]
Y_val = Y_test[:split_idx]
X_test = X_test[split_idx:]
Y_test = Y_test[split_idx:]

X_train = torch.FloatTensor(X_train).to(device)
Y_train = torch.FloatTensor(Y_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
Y_val = torch.FloatTensor(Y_val).to(device)
X_test = torch.FloatTensor(X_test).to(device)
Y_test = torch.FloatTensor(Y_test).to(device)

# create data loaders
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
test_dataset = TensorDataset(X_test, Y_test)

# Set determinism if needed
if args.seed >= 0:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True,
    drop_last=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False
)

import torch
import torch.nn as nn
import torch
import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self, input_length, classes, num_conv=2, num_linear=1, width=1.0, linear_dropout=0.5, noise_std = 0.2, growth_mode=0):
        """
        Args:
            input_length: Length of input audio features
            classes: Number of output classes
            num_conv: Number of convolutional blocks (0-3)
            num_linear: Number of linear layers (1-3)
            width: Width multiplier for channels (0.0625-8.0)
            linear_dropout: Dropout rate for linear layers (0.0-1.0)
        """
        super(AudioClassifier, self).__init__()
        
        # Validate parameters
        assert 0 <= num_conv <= 4, "num_conv must be between 0 and 4"
        assert 1 <= num_linear <= 2, "num_linear must be between 1 and 2"
        assert 0.0625 <= width <= 8.0, "width must be between 0.0625 and 8.0"
        assert 0.0 <= linear_dropout <= 1.0, "linear_dropout must be between 0.0 and 1.0"
        
        self.input_length = input_length
        self.classes = classes
        self.num_conv = num_conv
        self.num_linear = num_linear
        self.width = width
        self.linear_dropout = linear_dropout
        
        # Data augmentation - GaussianNoise
        self.noise_std = noise_std
        
        # Reshape parameters
        self.channels = 1
        self.columns = 13
        self.rows = int(input_length / (self.columns * self.channels))
        
        # Calculate channel sizes with width multiplier
        if(growth_mode == 0):
            base_channels = [8, 16, 32, 64]
        elif growth_mode == 1:
            base_channels = [8, 16, 24, 32]
        elif growth_mode == 2:
            base_channels = [8, 16, 16, 32]
        elif growth_mode == 3:
            base_channels = [8, 16, 16, 16]
        elif growth_mode == 4:
            base_channels = [8, 8, 8, 8]
        elif growth_mode == 5:
            base_channels = [8, 8, 8, 16]
        self.channel_sizes = [max(1, int(ch * width)) for ch in base_channels]
        
        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()
        # store conv layer refs for applying constraints outside forward
        self._constrained_conv_layers = []
        self.conv_hooks = []
        
        if num_conv > 0:
            in_channels = self.channels
            for i in range(num_conv):
                out_channels = self.channel_sizes[i]
                
                conv_block = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding='same'
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25)
                )
                
                self.conv_blocks.append(conv_block)
                
                # Keep reference to conv layer(s) so we can apply max-norm after optimizer.step()
                conv_layer = conv_block[0]
                self._constrained_conv_layers.append(conv_layer)
                
                # Do NOT mutate weights inside forward_pre_hook. Hooks that mutate parameters during forward
                # cause ONNX warnings because they produce in-place ops on block inputs.
                # If you previously relied on the hook to enforce max-norm, instead call enforce_max_norm()
                # after optimizer.step() in the training loop.
                
                in_channels = out_channels
        
        # Calculate flattened size after conv layers
        if num_conv > 0:
            final_rows = self.rows // (2 ** num_conv)
            final_cols = self.columns // (2 ** num_conv)
            flattened_size = self.channel_sizes[num_conv - 1] * final_rows * final_cols
        else:
            # No conv layers, input goes directly to linear layers
            flattened_size = input_length
        
        # Build linear layers
        self.linear_layers = nn.ModuleList()
        
        # Calculate linear layer sizes
        linear_sizes = self._calculate_linear_sizes(flattened_size, classes, num_linear)
        
        for i in range(num_linear - 1):
            self.linear_layers.append(nn.Sequential(
                nn.Dropout(linear_dropout),
                nn.Linear(linear_sizes[i], linear_sizes[i + 1]),
                nn.ReLU()
            ))
        
        # Final output layer (with dropout before it)
        self.linear_layers.append(nn.Sequential(
            nn.Dropout(linear_dropout),
            nn.Linear(linear_sizes[-2], linear_sizes[-1])
        ))
    
    def _calculate_linear_sizes(self, input_size, output_size, num_layers):
        """Calculate intermediate layer sizes for linear layers"""
        if num_layers == 1:
            return [input_size, output_size]
        
        # Create logarithmically spaced sizes from input to output
        sizes = [input_size]
        
        # Calculate intermediate sizes
        log_start = torch.log(torch.tensor(float(input_size)))
        log_end = torch.log(torch.tensor(float(output_size)))
        
        for i in range(1, num_layers):
            ratio = i / num_layers
            log_size = log_start + (log_end - log_start) * ratio
            size = max(output_size, int(torch.exp(log_size).item()))
            sizes.append(size)
        
        sizes.append(output_size)
        return sizes
    
    def _max_norm_constraint_tensor(self, weight: torch.Tensor, max_value=1.0) -> torch.Tensor:
        """Return a constrained weight tensor (out-of-place)."""
        norm = weight.norm(2, dim=(1, 2, 3), keepdim=True)
        desired = torch.clamp(norm, max=max_value)
        scale = (desired / (norm + 1e-8))
        return weight * scale

    def enforce_max_norm(self, max_value=1.0):
        """Apply max-norm constraint to stored conv layers OUTSIDE of forward.
        Call this after optimizer.step() during training.
        This mutates parameters but it happens outside forward, so ONNX export won't see in-forward mutations.
        """
        for module in self._constrained_conv_layers:
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    # Mutate here is acceptable because it's outside forward (post-step)
                    norm = module.weight.norm(2, dim=(1, 2, 3), keepdim=True)
                    desired = torch.clamp(norm, max=max_value)
                    module.weight *= (desired / (norm + 1e-8))

    def forward(self, x):
        # Add Gaussian noise during training
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Apply convolutional layers if any
        if self.num_conv > 0:
            # Reshape: (batch, input_length) -> (batch, channels, rows, columns)
            x = x.view(-1, self.channels, self.rows, self.columns)
            
            for conv_block in self.conv_blocks:
                x = conv_block(x)
            
            # Flatten
            x = x.view(x.size(0), -1)
        
        # Apply linear layers
        for linear_layer in self.linear_layers:
            x = linear_layer(x)
        
        return x

def main(config):
    # config is the argparse.Namespace (args)
    # Map improvement_threshold to internal thresholds
    if args.improvement_threshold == 'high':
        thresh = [0.01, 0.001, 0.0001, 0]
    elif args.improvement_threshold == 'medium':
        thresh = [0.001, 0.0001, 0]
    elif args.improvement_threshold == 'low':
        thresh = [0]
    GPA.pc.set_improvement_threshold(thresh)
    GPA.pc.set_candidate_weight_initialization_multiplier(
        args.dendrite_weight_initialization_multiplier
    )
    if args.dendrite_forward_function == 'sigmoid':
        pai_forward_function = torch.sigmoid
    elif args.dendrite_forward_function == 'relu':
        pai_forward_function = torch.relu
    elif args.dendrite_forward_function == 'tanh':
        pai_forward_function = torch.tanh
    GPA.pc.set_pai_forward_function(pai_forward_function)
    if args.dendrite_conversion == 'All Layers':
        GPA.pc.set_modules_to_convert([nn.Conv2d, nn.Linear])
        GPA.pc.set_modules_to_track([])
    elif args.dendrite_conversion == 'Linear Only':
        GPA.pc.set_modules_to_convert([nn.Linear])
        GPA.pc.set_modules_to_track([nn.Conv2d])
        
    if not str2bool(args.dendritic_optimization):
        print('building without dendrites')
        GPA.pc.set_max_dendrites(0)
    else:
        print('building with dendrites')
        GPA.pc.set_max_dendrites(args.max_dendrites)
    if not str2bool(args.improved_dendritic_optimization):
        GPA.pc.set_perforated_backpropagation(False)
        GPA.pc.set_dendrite_update_mode(True)
    else:
        print('building with improved dendrites')
        GPA.pc.set_perforated_backpropagation(True)
        GPA.pc.set_dendrite_update_mode(True)
    GPA.pc.set_initial_correlation_batches(40)

    # Initialize model
    input_length = 624
    classes = 3
    model = AudioClassifier(input_length, classes, num_conv=args.num_conv, num_linear=args.num_linear, width=args.network_width, linear_dropout=args.dropout, noise_std=args.noise_std, growth_mode=args.channel_growth_mode).to(device)

    GPA.pc.set_testing_dendrite_capacity(False)
    
    if args.switch_speed == 'fast':
        GPA.pc.set_n_epochs_to_switch(10)
    elif args.switch_speed == 'medium':
        GPA.pc.set_n_epochs_to_switch(25)
    elif args.switch_speed == 'slow':
        GPA.pc.set_n_epochs_to_switch(100)


    GPA.pc.set_verbose(False)
    GPA.pc.set_silent(True)

    model = UPA.initialize_pai(model)
    print(model)

    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    optimArgs = {'params':model.parameters(),
                 'lr':args.learning_rate,
                 'betas':(0.9, 0.999)}
    schedArgs = {'mode':'max', 'patience': 5} #Make sure this is lower than epochs to switch
    optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop helpers
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Convert one-hot labels to class indices if needed
            if labels.dim() > 1 and labels.size(1) > 1:
                labels = torch.argmax(labels, dim=1)
            
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Enforce max-norm constraint AFTER optimizer.step() to avoid in-forward mutation
            # (This mutates parameters but it's outside forward; ONNX exporter won't see it)
            if hasattr(model, 'enforce_max_norm'):
                model.enforce_max_norm(max_value=1.0)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def test(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                
                # Convert one-hot labels to class indices if needed
                if labels.dim() > 1 and labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)
                
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    first_test_loss = 0
    first_test_acc = 0
    first_val_loss = 0
    first_val_acc = 0
    first_param_count = UPA.count_params(model)
    # Training loop
    epoch = -1

    while True:
        epoch += 1
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = test(model, val_loader, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)


        GPA.pai_tracker.add_extra_score(train_acc, 'Train')
        GPA.pai_tracker.add_extra_score(test_acc, 'Test')
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
        model.to(device)
        if(training_complete):
            break
        elif(restructured):
            if(first_test_loss == 0):
                first_test_loss = test_loss
                first_test_acc = test_acc
                first_val_loss = val_loss
                first_val_acc = val_acc
                

            print('Restructured dendritic architecture')
            optimArgs = {'params':model.parameters(),
                        'lr':args.learning_rate,
                        'betas':(0.9, 0.999)}
            schedArgs = {'mode':'max', 'patience': 5}
            optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

        print(f'Epoch {epoch+1} '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | '
            f'Dendrite Count and Mode: {GPA.pai_tracker.member_vars["num_dendrites_added"]}'
            f' - {GPA.pai_tracker.member_vars["mode"]}')
    
    test_loss, test_acc = test(model, test_loader, criterion, device)
    if str2bool(args.dendritic_optimization):
        print(f'First architecture: '
                f'Val Acc: {first_val_acc:.4f}, Test Acc: {first_test_acc:.4f}, params: {first_param_count}')

        print(f'Final architecture: '
                f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, params: {UPA.count_params(model)} '
                f'Dendrite Count: {GPA.pai_tracker.member_vars["num_dendrites_added"]}')

        print(f'Reduction in misclassifications because of dendrites')
        print(f'Validation: {(100.0*((val_acc-first_val_acc)/(1-first_val_acc))):.2f}%')
        print(f'Test: {(100.0*((test_acc-first_test_acc)/(1-first_test_acc))):.2f}%')
    else:
        print(f'Final architecture: '
        f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, params: {UPA.count_params(model)} '
        f'Dendrite Count: {GPA.pai_tracker.member_vars["num_dendrites_added"]}')

    from perforatedbp import network_pbp as PBN
    model = AudioClassifier(input_length, classes, num_conv=args.num_conv, num_linear=args.num_linear, width=args.network_width, linear_dropout=args.dropout, noise_std=args.noise_std, growth_mode=args.channel_growth_mode).to(device)

    # If doing open source mode for now just do pai save for onnx
    if(not str2bool(args.improved_dendritic_optimization)):
        from perforatedbp import utils_pbp as PBU
        model = UPA.initialize_pai(model)
        model = UPA.load_system(model, 'PAI', 'best_model', True)
        PBU.pb_save_net(model,'PAI','best_model')
        model = AudioClassifier(input_length, classes, num_conv=args.num_conv, num_linear=args.num_linear, width=args.network_width, linear_dropout=args.dropout, noise_std=args.noise_std, growth_mode=args.channel_growth_mode).to(device)
        
    model = PBN.load_pai_model(model, 'PAI/best_model_pai.pt')
    for i, block in enumerate(model.conv_blocks):
        for conv in block[0].layer_array:
            if conv.padding == 'same':
                # Calculate explicit padding
                if isinstance(conv.kernel_size, tuple):
                    padding = tuple((k - 1) // 2 for k in conv.kernel_size)
                else:
                    padding = (conv.kernel_size - 1) // 2
                
                # Set the explicit padding
                conv.padding = padding

    # If some forward_pre_hooks still exist and might mutate weights, remove them before exporting as a quick fallback:
    # for h in getattr(model, 'conv_hooks', []):
    #     try:
    #         h.remove()
    #     except Exception:
    #         pass

    torch.onnx.export(model.cpu(),
                  torch.randn((32, 624)),
                  os.path.join(args.out_directory, 'model.onnx'),
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'])

if __name__ == "__main__":
    main(args)
