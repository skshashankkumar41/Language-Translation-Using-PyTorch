import torch 
import torch.nn as nn
import torch.optim as optim 
from dataloader import get_loader
from dataset import TranslateDataset
from model import Encoder, Decoder, Seq2Seq
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

torch.cuda.empty_cache()

num_epochs = 100
learning_rate = 0.001
batch_size = 32

trainLoader, valLoader, testLoader, trainDataset = get_loader(
    root_path = 'Language-Translation-Using-PyTorch/input/',
    batch_size = batch_size
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(trainDataset.eng_vocab)
input_size_decoder = len(trainDataset.hin_vocab)
output_size = len(trainDataset.hin_vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300

hidden_size = 512
num_layers = 1
enc_dropout = 0.5
dec_dropout = 0.5

step = 0 

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net, trainDataset.hin_vocab, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = trainDataset.eng_vocab.stoi["<PAD>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

sentence = "this percentage is even greater than the percentage in india"

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)
    
    model.eval()

    translated_sentence = translate_sentence(
        model, sentence,trainDataset.eng_vocab, trainDataset.hin_vocab, device, max_length=50
    )
    
    print(f"Original example sentence: \n {sentence}")
    print(f"Translated example sentence: \n {' '.join(translated_sentence)}")
    model.train()

    for batch_idx, (src,trg) in enumerate(trainLoader):
        inp_data = src.to(device)
        target = trg.to(device)
        output = model(inp_data, target)

        
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        if step % 50 == 0:
            print("LOSS - ",loss.item())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        step += 1

    model.eval()
    score = bleu(valLoader, model, trainDataset.eng_vocab, trainDataset.hin_vocab, device)
    print(f"Validation Bleu score {score*100:.2f}")

# model.eval()
# score = bleu(testLoader, model, trainDataset.eng_vocab, trainDataset.hin_vocab, device)
# print(f"Bleu score {score*100:.2f}")