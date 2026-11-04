import argparse
import numpy as np
import torch
import os
import wandb
from cs336_basics import model as model_impl


@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, device):
    """在验证集上计算 Loss"""
    model.eval()  # 切换到评估模式 (关闭 Dropout 等)
    inputs, targets = model_impl.run_get_batch(val_data, batch_size, context_length, device)
    logits = model(inputs)
    # 展平计算 Loss
    loss = model_impl.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    model.train()  # 切回训练模式
    return loss.item()


def training_loop():
    # 1. 初始化 WandB (请确保提前在终端运行了 wandb login)
    wandb.init(project="cs336-lab1", name="transformer-tinystories")

    parser = argparse.ArgumentParser(description="CS336 Transformer Training")
    
    # --- 模型参数 ---
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--context_length", type=int, default=256)
    # 提醒：vocab_size 必须和你 BPE 训练出来的词表大小一致 (TinyStories 是 10000)
    parser.add_argument("--vocab_size", type=int, default=10000) 
    parser.add_argument("--rope_theta", type=float, default=10000.0) # 修复：添加了默认值
    
    # --- 训练参数 ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    
    # --- 路径参数 ---
    parser.add_argument("--train_data", type=str, default="train_tokens.npy")
    parser.add_argument("--val_data", type=str, default="valid_tokens.npy")

    args = parser.parse_args()
    
    # 将 argparse 的 Namespace 转为普通的字典，方便解包
    args_dict = vars(args)
    
    # 将超参数记录到 wandb 面板中，方便以后对比实验
    wandb.config.update(args_dict)

    # 2. 加载数据 (mmap 懒加载，不占内存)
    print(f"Loading training data from {args.train_data}...")
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")

    # 3. 初始化模型和优化器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 【重点】使用字典解包传给模型，彻底避免参数顺序传错的风险！
    # 提取属于模型的参数
    model_kwargs = {
        "vocab_size": args_dict["vocab_size"],
        "context_length": args_dict["context_length"],
        "d_model": args_dict["d_model"],
        "num_layers": args_dict["num_layers"],
        "num_heads": args_dict["num_heads"],
        "d_ff": args_dict["d_ff"],
        "rope_theta": args_dict["rope_theta"]
    }
    
    print("Initializing model...")
    model = model_impl.MyTransformerLm(**model_kwargs).to(device)
    
    # 优化器
    optimizer = model_impl.MyAdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Start training for {args.total_steps} steps on {device}...")

    # 4. 核心训练循环
    for step in range(args.total_steps):
        # A. 动态更新学习率 (余弦退火)
        # 假设 warmup 步数为 100
        lr = model_impl.get_lr_cosine_schedule(
            step, 
            alpha_max=args.lr, 
            alpha_min=args.lr * 0.1, 
            Tw=100, 
            Tc=args.total_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
            
        # B. 采样批次数据
        inputs, targets = model_impl.run_get_batch(
            train_data, args.batch_size, args.context_length, device
        )
        
        # C. 前向传播 & 计算 Loss
        logits = model(inputs)
        loss = model_impl.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # D. 反向传播 & 梯度裁剪
        optimizer.zero_grad()
        loss.backward()
        model_impl.clip_gradients(model.parameters(), max_norm=1.0)
        
        # E. 更新参数
        optimizer.step()
        
        # F. 记录日志 & 验证
        if step % 100 == 0:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device)
            print(f"Step {step:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.5f}")
            
            # 记录到 WandB
            wandb.log({
                "step": step,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "learning_rate": lr
            })
            
        # G. 保存检查点
        if step % 1000 == 0 and step > 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{step}.pt")
            model_impl.save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    # 训练结束后，保存最后一次的模型
    final_ckpt_path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    model_impl.save_checkpoint(model, optimizer, args.total_steps, final_ckpt_path)
    print(f"Training completed! Final model saved to {final_ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    training_loop()