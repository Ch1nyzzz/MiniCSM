from tqdm import tqdm

def train(args, student_model, teacher_model, train_dataloader, eval_dataloader, distill_utils, optimizer, scheduler):
    for epoch in range(args.num_train_epochs):
        # 训练阶段
        student_model.train()
        distill_utils.encoder_proj.train()
        distill_utils.decoder_proj.train()

        for step, (model_batch, no_model_batch) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            student_outputs = student_model(
                **model_batch,
                labels=no_model_batch["labels"],
                output_hidden_states=True
            )

            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **model_batch,
                    labels=no_model_batch["labels"],
                    output_hidden_states=True
                )

            loss = distill_utils.get_distil_loss(
                teacher_outputs,
                student_outputs,
                no_model_batch["labels"],
                temp=args.temp,
                kd_ratio=args.kd_ratio,
                inter_loss_weight=args.intermediate_layer_distil_weight,
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

        # 评估阶段
        eval_loss, eval_accuracy = evaluate(
            student_model, eval_dataloader
        )
        print(f"Epoch {epoch+1} Eval Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")

def evaluate(student_model, eval_dataloader):
    student_model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for model_batch, no_model_batch in tqdm(eval_dataloader, desc="Evaluation"):
            outputs = student_model(**model_batch, labels=no_model_batch["labels"])
            total_loss += outputs.loss.item()

            preds = student_model.generate(
                input_ids=model_batch["input_ids"],
                attention_mask=model_batch["attention_mask"],
                max_length=5
            )
            predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = tokenizer.batch_decode(no_model_batch["labels"], skip_special_tokens=True)

            correct += sum(1 for p, l in zip(predictions, labels) if p.strip() == l.strip())
            total += len(labels)

    avg_loss = total_loss / len(eval_dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy
