from torchmetrics.classification import MultilabelF1Score
f1 = MultilabelF1Score(num_labels=18, threshold=0.5, average='micro')
for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)

    F1= f1(torch.Tensor(outputs), torch.Tensor(targets))

    print(outputs, targets)

    print(F1)

    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")