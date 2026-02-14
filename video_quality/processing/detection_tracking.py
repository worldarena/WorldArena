from collections import defaultdict
import numpy as np
import os
import av
import cv2
import yaml
import torch
from tqdm import tqdm
from PIL import Image
import math

# ===== 1. SAM3导入 =====
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def nms_boxes(boxes, scores, iou_threshold=0.5):
    """
    非极大值抑制，移除重叠的检测框
    """
    if len(boxes) == 0:
        return [], []
    
    # 转换为numpy数组
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    
    # 按置信度排序
    indices = np.argsort(scores_np)[::-1]
    
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # 计算IoU
        current_box = boxes_np[current]
        other_boxes = boxes_np[indices[1:]]
        
        # 计算交集
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算并集
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_other = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union = area_current + area_other - intersection
        
        iou = intersection / (union + 1e-6)
        
        # 保留IoU小于阈值的框
        indices = indices[1:][iou < iou_threshold]
    
    return boxes_np[keep].tolist(), scores_np[keep].tolist()


class GripperDetector:
    """机械爪检测器，只加载一次模型"""
    
    def __init__(self, model_path=''):
        print("加载SAM3模型...")
        try:
            # 只加载一次模型，后续重复使用
            self.model = build_sam3_image_model(
                checkpoint_path=os.path.join(model_path, "sam3.pt"),
                bpe_path=os.path.join(model_path, "bpe_simple_vocab_16e6.txt.gz")
            )
            self.processor = Sam3Processor(self.model)
            print("SAM3模型加载成功")
        except Exception as e:
            print(f"加载SAM3模型失败: {e}")
            raise
    
    def detect_frame_single_prompt(self, image, prompt="end effector"):
        """
        使用单个提示词检测单张图像
        Args:
            image: PIL.Image 或 numpy array
            prompt: 文本提示词
        Returns:
            boxes: 检测框列表 [[x1,y1,x2,y2], ...]
            scores: 置信度列表
        """
        # 转换图像格式
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # BGR转RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        try:
            # 设置图像并进行编码
            inference_state = self.processor.set_image(image)
            
            # 设置文本提示
            inference_state = self.processor.set_text_prompt(
                state=inference_state, 
                prompt=prompt
            )
            
            # 获取检测结果
            boxes = inference_state["boxes"]
            scores = inference_state["scores"]
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            return boxes, scores
            
        except Exception as e:
            print(f"检测失败: {e}")
            return [], []
    
    def detect_frame(self, image, prompts=None):
        """
        使用多个提示词检测单张图像，并应用NMS
        Args:
            image: PIL.Image 或 numpy array
            prompts: 提示词列表，默认为 ["robot arm", "end effector", "gripper"]
        Returns:
            boxes: 检测框列表 [[x1,y1,x2,y2], ...]
            scores: 置信度列表
        """
        if prompts is None:
            prompts = ["robot arm", "end effector", "gripper"]
        
        all_boxes = []
        all_scores = []
        
        # 对每个提示词进行检测
        for prompt in prompts:
            try:
                boxes, scores = self.detect_frame_single_prompt(image, prompt)
                if len(boxes) > 0:
                    all_boxes.extend(boxes)
                    all_scores.extend(scores)
            except Exception as e:
                print(f"提示词 '{prompt}' 检测失败: {e}")
                continue
        
        # 如果检测到结果，应用NMS
        if len(all_boxes) > 0:
            return nms_boxes(all_boxes, all_scores, iou_threshold=0.3)
        
        return [], []
    
    def detect_batch(self, images, prompts=None):
        """
        批量检测多张图像（更高效）
        Args:
            images: 图像列表 [PIL.Image, ...]
            prompts: 提示词列表
        Returns:
            results: 检测结果列表 [(boxes, scores), ...]
        """
        results = []
        for img in tqdm(images, desc="批量检测", unit="帧"):
            boxes, scores = self.detect_frame(img, prompts)
            results.append((boxes, scores))
        return results


def check_if_already_processed(output_video_path, output_traj_path, input_path):
    """
    检查是否已经处理过该视频
    Returns:
        True: 已经处理完成
        False: 需要处理
    """
    # 检查输出目录是否存在
    if not os.path.exists(output_video_path) or not os.path.exists(output_traj_path):
        return False
    
    # 检查轨迹文件是否存在且有效
    traj_file = os.path.join(output_traj_path, 'traj.npy')
    if not os.path.exists(traj_file):
        return False
    
    # 检查轨迹文件是否有效（形状正确）
    try:
        traj_data = np.load(traj_file)
        if len(traj_data.shape) != 3 or traj_data.shape[1] != 2 or traj_data.shape[2] != 2:
            print(f"警告: 轨迹文件形状不正确: {traj_data.shape}")
            return False
    except Exception as e:
        print(f"警告: 无法加载轨迹文件: {e}")
        return False
    
    # 检查视频文件是否存在
    video_file = os.path.join(output_video_path, "video.mp4")
    if not os.path.exists(video_file):
        return False
    
    # 获取输入图像数量
    image_files = sorted([
        f for f in os.listdir(input_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # 检查轨迹数据帧数是否与输入图像数量匹配
    if len(traj_data) != len(image_files):
        print(f"警告: 轨迹帧数({len(traj_data)})与图像数量({len(image_files)})不匹配")
        return False
    
    print(f"✓ 已处理: {input_path}")
    return True


def interpolate_trajectory(traj_list, max_gap=5):
    """
    对轨迹进行插值，填补缺失的检测点
    Args:
        traj_list: 轨迹列表，每个元素是(x, y)坐标或(-1, -1)表示缺失
        max_gap: 最大允许的缺失间隔，超过此间隔不进行插值
    Returns:
        插值后的轨迹列表
    """
    if not traj_list:
        return traj_list
    
    traj_array = np.array(traj_list)
    result = traj_array.copy()
    
    # 找到所有有效点（非-1的点）
    valid_mask = np.all(traj_array != -1, axis=1)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 2:
        return traj_list
    
    # 对每个缺失段进行插值
    for i in range(len(valid_indices) - 1):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i + 1]
        gap = end_idx - start_idx - 1
        
        # 如果缺失间隔在允许范围内，进行线性插值
        if 0 < gap <= max_gap:
            start_point = traj_array[start_idx]
            end_point = traj_array[end_idx]
            
            # 线性插值
            for j in range(1, gap + 1):
                alpha = j / (gap + 1)
                interpolated_point = start_point * (1 - alpha) + end_point * alpha
                result[start_idx + j] = interpolated_point
    
    return result.tolist()


def smooth_trajectory(traj_list, window_size=3):
    """
    对轨迹进行平滑处理
    Args:
        traj_list: 轨迹列表
        window_size: 滑动窗口大小
    Returns:
        平滑后的轨迹列表
    """
    if len(traj_list) < window_size:
        return traj_list
    
    traj_array = np.array(traj_list)
    result = traj_array.copy()
    
    # 滑动平均平滑
    for i in range(len(traj_array)):
        start = max(0, i - window_size // 2)
        end = min(len(traj_array), i + window_size // 2 + 1)
        
        # 只使用有效点进行平均
        window_points = traj_array[start:end]
        valid_points = window_points[np.all(window_points != -1, axis=1)]
        
        if len(valid_points) > 0:
            result[i] = np.mean(valid_points, axis=0)
    
    return result.tolist()


def process_video_with_tracking(input_path, output_path, detector=None, gid=None,
                                data_type='val', saving_fps=12, force_reprocess=False):
    """
    使用SAM3检测机械爪并生成轨迹
    Args:
        detector: 已加载的GripperDetector实例
        force_reprocess: 强制重新处理，即使已有结果
    """
    
    print(f"处理: {input_path}")
    
    if detector is None:
        print("错误: 需要提供已加载的检测器")
        return False
    
    # ===== 创建输出目录 =====
    if data_type == 'val':
        output_video_path = os.path.join(output_path, gid, "gripper_detection")
        output_traj_path = os.path.join(output_path, gid, "traj")
    else:
        output_video_path = os.path.join(output_path, "gripper_detection")
        output_traj_path = os.path.join(output_path, "traj")
    
    os.makedirs(output_video_path, exist_ok=True)
    os.makedirs(output_traj_path, exist_ok=True)
    
    # ===== 检查是否已经处理过 =====
    if not force_reprocess and check_if_already_processed(output_video_path, output_traj_path, input_path):
        return True

    # ===== 准备输出视频 =====
    output_video_file = os.path.join(output_video_path, "video.mp4")
    output_container = av.open(output_video_file, mode='w', format='mp4')
    output_stream = output_container.add_stream('h264', rate=saving_fps)
    output_stream.width = 640
    output_stream.height = 480
    output_stream.pix_fmt = 'yuv420p'

    # ===== 轨迹数据存储 =====
    trajectory_data = []
    
    # 使用列表存储历史轨迹点，而不是defaultdict
    left_track_history = []  # 左爪轨迹历史
    right_track_history = []  # 右爪轨迹历史
    
    # 存储最后有效的检测点
    last_valid_left = None
    last_valid_right = None
    
    # 记录连续缺失帧数
    left_missing_count = 0
    right_missing_count = 0
    
    # 最大允许缺失帧数，超过则不使用历史点
    MAX_MISSING_FRAMES = 10

    # ===== 获取图像文件列表 =====
    image_files = sorted([
        f for f in os.listdir(input_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    if not image_files:
        print(f"警告: {input_path} 中没有图像文件")
        output_container.close()
        return False
    
    print(f"找到 {len(image_files)} 张图像，开始处理...")
    
    # ===== 预加载所有图像到内存（可选的优化） =====
    print("加载图像到内存...")
    images = []
    for img_file in tqdm(image_files, desc="加载图像", unit="帧"):
        img_path = os.path.join(input_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
            images.append(img)
        else:
            images.append(None)  # 用None占位
    
    # ===== 批量检测（可选）或逐帧检测 =====
    print("开始检测...")
    for global_frame_idx in tqdm(range(len(images)), desc="处理帧", unit="帧"):
        img = images[global_frame_idx]
        
        if img is None:
            # 如果读取失败，用(-1,-1)填充两个爪
            left_current = (-1, -1)
            right_current = (-1, -1)
            
            # 添加到轨迹数据
            trajectory_data.append([list(left_current), list(right_current)])
            
            # 创建空白帧
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            output_frame = av.VideoFrame.from_ndarray(blank_frame, format='rgb24')
            for packet in output_stream.encode(output_frame):
                output_container.mux(packet)
            continue
        
        # 准备可视化帧
        annotated_frame = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        
        # ===== 使用检测器检测当前帧 =====
        # 使用多提示词检测
        boxes, scores = detector.detect_frame(img, prompts=["robot arm", "end effector", "gripper"])
        
        # ===== 解析检测结果 =====
        left_detections = []  # 左爪候选
        right_detections = []  # 右爪候选
        
        # 处理每个检测框
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score < 0.3:  # 置信度阈值
                continue
            
            # 确保box是列表
            if isinstance(box, (np.ndarray, torch.Tensor)):
                box = box.tolist() if hasattr(box, 'tolist') else list(box)
            
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            
            # 计算归一化坐标
            x_norm = x_center / 640.0
            y_norm = y_center / 480.0
            
            # 根据x坐标判断左右爪（以图像中心为界）
            if x_center < 320:  # 左半边
                left_detections.append({
                    'box': box,
                    'score': score,
                    'center': (x_center, y_center),
                    'norm': (x_norm, y_norm),
                    'index': i
                })
            else:  # 右半边
                right_detections.append({
                    'box': box,
                    'score': score,
                    'center': (x_center, y_center),
                    'norm': (x_norm, y_norm),
                    'index': i
                })
        
        # ===== 处理左爪检测 =====
        left_current = None
        
        if left_detections:
            # 检测到左爪，选择置信度最高的
            best_left = max(left_detections, key=lambda x: x['score'])
            x_norm, y_norm = best_left['norm']
            x_center, y_center = best_left['center']
            
            left_current = (float(x_norm), float(y_norm))
            last_valid_left = left_current
            left_missing_count = 0
            
            # 可视化左爪（红色框）
            x1, y1, x2, y2 = map(int, best_left['box'])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 标签：显示置信度
            label = f"Left: {best_left['score']:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif last_valid_left is not None and left_missing_count < MAX_MISSING_FRAMES:
            # 未检测到左爪，但之前有有效点且在允许范围内
            left_current = last_valid_left
            left_missing_count += 1
        else:
            # 完全缺失
            left_current = (-1.0, -1.0)
            left_missing_count += 1
        
        # ===== 处理右爪检测 =====
        right_current = None
        
        if right_detections:
            # 检测到右爪，选择置信度最高的
            best_right = max(right_detections, key=lambda x: x['score'])
            x_norm, y_norm = best_right['norm']
            x_center, y_center = best_right['center']
            
            right_current = (float(x_norm), float(y_norm))
            last_valid_right = right_current
            right_missing_count = 0
            
            # 可视化右爪（绿色框）
            x1, y1, x2, y2 = map(int, best_right['box'])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 标签：显示置信度
            label = f"Right: {best_right['score']:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif last_valid_right is not None and right_missing_count < MAX_MISSING_FRAMES:
            # 未检测到右爪，但之前有有效点且在允许范围内
            right_current = last_valid_right
            right_missing_count += 1
        else:
            # 完全缺失
            right_current = (-1.0, -1.0)
            right_missing_count += 1
        
        # ===== 更新轨迹历史 =====
        if left_current != (-1, -1):
            # 转换回像素坐标用于可视化
            x_pixel = left_current[0] * 640
            y_pixel = left_current[1] * 480
            left_track_history.append((int(x_pixel), int(y_pixel)))
        else:
            # 如果当前帧缺失，保留历史轨迹但不添加新点
            pass
        
        if right_current != (-1, -1):
            # 转换回像素坐标用于可视化
            x_pixel = right_current[0] * 640
            y_pixel = right_current[1] * 480
            right_track_history.append((int(x_pixel), int(y_pixel)))
        else:
            # 如果当前帧缺失，保留历史轨迹但不添加新点
            pass
        
        # ===== 绘制轨迹线 =====
        # 左爪轨迹（红色）
        if len(left_track_history) > 1:
            # 使用更粗的线条绘制轨迹
            for i in range(1, len(left_track_history)):
                cv2.line(annotated_frame, 
                        left_track_history[i-1], 
                        left_track_history[i], 
                        (255, 0, 0),  # BGR格式的红色
                        3,  # 线条粗细
                        cv2.LINE_AA)  # 抗锯齿
        
        # 右爪轨迹（红色）
        if len(right_track_history) > 1:
            # 使用更粗的线条绘制轨迹
            for i in range(1, len(right_track_history)):
                cv2.line(annotated_frame, 
                        right_track_history[i-1], 
                        right_track_history[i], 
                        (255, 0, 0),  # BGR格式的红色
                        3,  # 线条粗细
                        cv2.LINE_AA)  # 抗锯齿
        
        # ===== 保存轨迹数据 =====
        trajectory_data.append([list(left_current), list(right_current)])
        
        # # ===== 添加帧编号和轨迹长度信息 =====
        # # 在左上角显示帧编号
        # cv2.putText(annotated_frame, f"Frame: {global_frame_idx}", (10, 30),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # # 显示轨迹长度
        # cv2.putText(annotated_frame, f"Left Traj: {len(left_track_history)}", (10, 60),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.putText(annotated_frame, f"Right Traj: {len(right_track_history)}", (10, 80),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # ===== 写入视频帧 =====
        output_frame = av.VideoFrame.from_ndarray(annotated_frame, format='rgb24')
        for packet in output_stream.encode(output_frame):
            output_container.mux(packet)
    
    # ===== 完成视频写入 =====
    for packet in output_stream.encode():
        output_container.mux(packet)
    output_container.close()
    
    # ===== 后处理轨迹数据（插值和平滑） =====
    print("后处理轨迹数据...")
    
    # 分离左右轨迹
    left_traj = [point[0] for point in trajectory_data]
    right_traj = [point[1] for point in trajectory_data]
    
    # 插值填补缺失点
    left_traj_interp = interpolate_trajectory(left_traj, max_gap=5)
    right_traj_interp = interpolate_trajectory(right_traj, max_gap=5)
    
    # 平滑轨迹
    left_traj_smooth = smooth_trajectory(left_traj_interp, window_size=3)
    right_traj_smooth = smooth_trajectory(right_traj_interp, window_size=3)
    
    # 重新组合轨迹数据
    final_trajectory_data = []
    for i in range(len(left_traj_smooth)):
        final_trajectory_data.append([list(left_traj_smooth[i]), list(right_traj_smooth[i])])
    
    trajectory_array = np.array(final_trajectory_data, dtype=np.float32)
    
    print(f"轨迹数据形状: {trajectory_array.shape}")
    print(f"有效左爪点: {np.sum(np.all(trajectory_array[:, 0] != -1, axis=1))}")
    print(f"有效右爪点: {np.sum(np.all(trajectory_array[:, 1] != -1, axis=1))}")
    
    # 保存文件
    output_file = os.path.join(output_traj_path, 'traj.npy')
    np.save(output_file, trajectory_array)
    print(f"轨迹已保存: {output_file}")
    
    # # 同时保存原始轨迹（可选）
    # raw_trajectory_array = np.array(trajectory_data, dtype=np.float32)
    # raw_output_file = os.path.join(output_traj_path, 'traj_raw.npy')
    # np.save(raw_output_file, raw_trajectory_array)
    
    return True


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, 
                       help='Path to config.yaml')
    parser.add_argument('--detect_gt', action='store_false',
                       help='是否检测ground truth轨迹')
    parser.add_argument('--force_reprocess', action='store_true',
                       help='强制重新处理所有视频，即使已有结果')
    args = parser.parse_args()
    
    config = load_config(args.config_path)
    
    # 从config读取路径
    data_base = config['data']['val_base']
    gt_path = config['data']['gt_path']
    
    # 模型路径
    model_path = config.get('ckpt', {}).get('sam3_model_ckpt', '')
    
    print("=" * 60)
    print("SAM3机械爪检测轨迹生成（改进版）")
    print("改进功能：")
    print("1. 轨迹中断时使用历史点填充")
    print("2. 轨迹线颜色改为红色")
    print("3. 后处理插值和平滑")
    print(f"强制重新处理: {'是' if args.force_reprocess else '否'}")
    print("=" * 60)
    
    # ===== 只加载一次模型 =====
    print("初始化检测器...")
    try:
        detector = GripperDetector(model_path=model_path)
    except Exception as e:
        print(f"初始化检测器失败: {e}")
        exit(1)
    
    # 统计处理结果
    processed_count = 0
    skipped_count = 0
    total_count = 0
    
    # 遍历所有任务
    for task in sorted(os.listdir(data_base)):
        task_path = os.path.join(data_base, task)
        
        if not os.path.isdir(task_path):
            continue
            
        for episode in sorted(os.listdir(task_path)):
            if episode.endswith(('.png', '.json')):
                continue
                
            episode_path = os.path.join(task_path, episode)
            
            # 处理ground truth视频
            if args.detect_gt:
                gt_episode_path = os.path.join(gt_path, task, episode)
                gt_video = os.path.join(gt_episode_path, 'video')
                
                if os.path.exists(gt_video):
                    total_count += 1
                    print(f"\n[处理GT] 任务: {task}, 片段: {episode}")
                    if process_video_with_tracking(
                        input_path=gt_video,
                        output_path=gt_episode_path,
                        detector=detector,
                        gid=None,
                        data_type='gt',
                        force_reprocess=args.force_reprocess
                    ):
                        processed_count += 1
                    else:
                        skipped_count += 1
            
            # 处理生成的视频
            for gid in sorted(os.listdir(episode_path)):
                input_path = os.path.join(episode_path, gid, "video")
                
                if os.path.exists(input_path):
                    total_count += 1
                    print(f"\n[处理生成] 任务: {task}, 片段: {episode}, GID: {gid}")
                    if process_video_with_tracking(
                        input_path=input_path,
                        output_path=episode_path,
                        detector=detector,
                        gid=gid,
                        data_type='val',
                        force_reprocess=args.force_reprocess
                    ):
                        processed_count += 1
                    else:
                        skipped_count += 1
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"总视频数: {total_count}")
    print(f"新处理视频: {processed_count}")
    print(f"跳过视频: {skipped_count}")
    print("=" * 60)