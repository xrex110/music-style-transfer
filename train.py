from torch.autograd import Variable
from utils import *
from models import *
import time
import math

cuda = False
if torch.cuda.is_available():
    cuda = True
else:
    cuda = False

print(f"Using gpu: {cuda}")

#Params
content_file = "input_stairway.wav"
style_file = "input_nightcall.wav"

style_param = 1
content_param = 1e2 

lr = 0.003
num_epochs = 20000

a_content, sr = wav2spectrum(content_file)
a_style, sr = wav2spectrum(style_file)

a_content_torch = torch.from_numpy(a_content)[None, None, :, :]
if cuda:
    a_content_torch = a_content_torch.cuda()
print(f"Shape on content file is : {a_content_torch.shape}")
a_style_torch = torch.from_numpy(a_style)[None, None, :, :]
if cuda:
    a_style_torch = a_style_torch.cuda()
print(f"Shape on style file is : {a_style_torch.shape}")

model = MusicCNN()
print(f"Model used:")
model.eval()

a_C_var = Variable(a_content_torch, requires_grad = False).float()
a_S_var = Variable(a_style_torch, requires_grad = False).float()

if cuda:
    model = model.cuda()
    a_C_var = a_C_var.cuda()
    a_S_var = a_S_var.cuda()

a_C = model(a_C_var)
a_S = model(a_S_var)



a_G_var = Variable(torch.randn(a_content_torch.shape) * 1e-3)
if cuda:
    a_G_var = a_G_var.cuda()
a_G_var.requires_grad = True
optimizer = torch.optim.Adam([a_G_var])


# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()
# Train the Model
for epoch in range(1, num_epochs + 1):
    print(f"Running epoch {epoch}")
    optimizer.zero_grad()
    a_G = model(a_G_var)

    content_loss = content_param * compute_content_loss(a_C, a_G)
    style_loss = style_param * compute_layer_style_loss(a_S, a_G)
    loss = content_loss + style_loss
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
        gen_audio_C = "out" + str(epoch) + ".wav"
        spectrum2wav(gen_spectrum, sr, gen_audio_C)    
    
    # print
    print("\t{} {}% {} content_loss:{:4f} style_loss:{:4f} total_loss:{:4f}".format(epoch,
                                                                                    epoch / num_epochs * 100,
                                                                                    timeSince(start),
                                                                                    content_loss.item(),
                                                                                    style_loss.item(), loss.item()))
    current_loss += loss.item()

gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
gen_audio_C = "out" + ".wav"
spectrum2wav(gen_spectrum, sr, gen_audio_C)