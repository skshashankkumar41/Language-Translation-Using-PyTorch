import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys


def translate_sentence(model, sentence, english, hindi, device, max_length=50):
    # print(sentence)

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.lower() for token in sentence.split(" ")]
        tokens.insert(0, '<SOS>')
        tokens.append('<EOS>')

        # Go through each german token and convert to an index
        text_to_indices = [english.stoi[token] if token in english.stoi else english.stoi["<UNK>"] for token in tokens]

        # Convert to Tensor
        sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    else:
        sentence_tensor = torch.LongTensor(sentence).unsqueeze(1).to(device)

    # print(tokens)

    # Add <SOS> and <EOS> in beginning and end respectively
    
    # Build encoder hidden, cell state
    with torch.no_grad():
        encoder_states, hidden, cell = model.encoder(sentence_tensor)

    outputs = [hindi.stoi["<SOS>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word,encoder_states, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == hindi.stoi["<EOS>"]:
            break

    translated_sentence = [hindi.itos[idx] if idx in hindi.itos else hindi.itos[3] for idx in outputs]

    # remove start token
    return translated_sentence[1:-1]


def bleu(data, model, english, hindi, device):
    targets = []
    outputs = []

    for (src,trg) in data:
        for src_text, trg_text in zip(src.permute(1,0),trg.permute(1,0)):
            trg_text = [hindi.itos[idx.item()] if idx.item() in hindi.itos else hindi.itos[3] for idx in trg_idx]
            trg_text = [i for i in trg_text if i != '<PAD>']

            prediction = translate_sentence(model, src_text, english, hindi, device)
            prediction = prediction[:-1]  # remove <eos> token

            targets.append([trg_text])
            outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="Language-Translation-Using-PyTorch/output/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])