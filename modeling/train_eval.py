import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, train_dataloader, val_dataloader, save_path, epoches = 10, learning_rate = 0.001, device = 'cpu', patience = 10, lr_patience = 5):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode = 'min', factor = 0.5, patience = lr_patience
    )

    history_train_loss = []
    history_train_acc = []
    history_val_loss = []
    history_val_acc = []


    best_val_loss = float('inf')
    patience_counter = 0


    
    
    for epoch in range(epoches):
        print(f"\nEpoch {epoch+1}/{epoches}")
        print("-" * 30)

        # ~~~~~~Train

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_dataloader, desc='Training', leave=False)

        for batch_features, batch_labels in train_pbar:
            #Device
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            #Forward
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            #Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_features.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

            current_acc = train_correct / train_total
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        
        # ============ Validation Phase ============
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(val_dataloader, desc='Validation', leave=False)
        
        with torch.no_grad():
            for batch_features, batch_labels in val_pbar:
                # Move to device
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device).squeeze()
                
                # Forward pass
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # Track metrics
                val_loss += loss.item() * batch_features.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

                current_acc = val_correct / val_total
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
        
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        
        history_train_loss.append(epoch_train_loss)
        history_train_acc.append(epoch_train_acc)
        history_val_loss.append(epoch_val_loss)
        history_val_acc.append(epoch_val_acc)
        
     
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")

        #lr
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {current_lr:.6f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save(model.state_dict(), save_path)
            print(f"New best model: Val Loss: {best_val_loss:.4f} Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epoches')
            
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch + 1}')
                #print(f'Best: Loss {best_val_loss:.4f}, Acc {best_val_acc:.4f} at epoch {best_epoch}')
                break
    
    #best model
    model.load_state_dict(torch.load(save_path))
    print(f'\nLoaded best model from epoch {best_epoch}')

    print('\nTraining complete')
    print(f'Best Val Loss: {best_val_loss:.4f}')
    print(f'Best Val Acc: {best_val_acc:.4f}')

    return history_train_loss, history_train_acc, history_val_loss, history_val_acc





def evaluate_model(model, test_loader, label_mapping, device='cpu'):
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []


    with torch.no_grad():
        for batch_features, batch_labels in tqdm(test_loader, desc='Testing'):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1-Score:  {f1:.4f}')



    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2%}')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


    print('Per class breakdown:')
    target_names_with_nums = [f"{label} ({num})" for label, num in label_mapping.items()]
    print(classification_report(all_labels, all_preds, 
                            target_names=target_names_with_nums))
    
    
    return accuracy, precision, recall, f1
