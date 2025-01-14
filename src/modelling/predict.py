def predict(self, model, data):
        import glob

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint_files = glob.glob(f'checkpoints/{self.img_size}_checkpoint_epoch_*.pt')

        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError("No checkpoint files found")
        
        model.to(device)
        model.eval()
        with torch.no_grad():
            for data, _ in data:
                #print(data)
                data = data.to(device)
                input_img = data[:, 0, :, :, :]
                fused_img = data[:, -1, :, :, :]

                #batch = next(iter(data))[0].to(device)
                output_img = model(input_img)

                output_img = self.normalize_to_target(output_img, fused_img)

                break

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(input_img.cpu().squeeze().numpy(), cmap='gray')
        axes[0].axis('off')

        axes[1].imshow(fused_img.cpu().squeeze().numpy(), cmap='gray')
        axes[1].axis('off')

        axes[2].imshow(output_img.cpu().squeeze().numpy(), cmap='gray')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        return output_img