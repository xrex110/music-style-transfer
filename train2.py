from torch.autograd import Variable
from utils import *
from models import *
import time
import math

LOGGING_INTERVAL = 1000
PRINTING_INTERVAL = 1

PREPROCESS_STYLE_TEMPO = False

cuda = False
if torch.cuda.is_available():
    cuda = True
else:
    cuda = False

print(f"Using gpu: {cuda}")

#Params
content_file = "input_stairway.wav"
style_file = "input_nightcall.wav"

content_wav, sr_content = librosa.load(content_file)

content_wav = np.split(content_wav, 2)[0]


if PREPROCESS_STYLE_TEMPO:
    style_wav, sr_style = librosa.load(style_file)
    paced_style_wav = changeOutputTempo(style_wav, content_wav, sr_style)
else:
    paced_style_wav, sr_style = librosa.load(style_file)

paced_style_wav = np.split(paced_style_wav, 2)[0]

style_param = 1
content_param = 1e2 

lr = 0.003
num_epochs = 20000

a_content, sr = fileToSpectrum(content_wav, sr_content)
a_style, sr = fileToSpectrum(paced_style_wav, sr_style)

a_content_torch = torch.from_numpy(a_content)[None, None, :, :]
if cuda:
    a_content_torch = a_content_torch.cuda()
print(f"Shape on content file is : {a_content_torch.shape}")
a_style_torch = torch.from_numpy(a_style)[None, None, :, :]
if cuda:
    a_style_torch = a_style_torch.cuda()
print(f"Shape on style file is : {a_style_torch.shape}")

model = ESCModel3()
model.load_state_dict(torch.load("esc50-model.pt")) #Load weights
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
    optimizer.zero_grad()
    a_G = model.feature_extractor(a_G_var)

    content_loss = content_param * compute_content_loss(a_C, a_G)
    style_loss = style_param * compute_layer_style_loss(a_S, a_G)
    loss = content_loss + style_loss
    loss.backward()
    optimizer.step()

    if epoch % PRINTING_INTERVAL == 0:
        print(f"Running epoch {epoch}")
        # print
        print("\t{} {:.2f}% {} content_loss:{:.4f} style_loss:{:.4f} total_loss:{:.4f}".format(epoch,
            epoch / num_epochs * 100, timeSince(start), content_loss.item(), style_loss.item(), loss.item()))

    if epoch % LOGGING_INTERVAL == 0:
        gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
        gen_audio_C = "out" + str(int(epoch / LOGGING_INTERVAL)) + ".wav"
        spectrum2wav(gen_spectrum, sr, gen_audio_C)

    current_loss += loss.item()

gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
gen_audio_C = "out" + ".wav"
spectrum2wav(gen_spectrum, sr, gen_audio_C)