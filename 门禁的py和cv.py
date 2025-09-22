# 在文件开头添加cv2导入
import cv2
import numpy as np
import os
import time
from datetime import datetime
import pickle
import logging

# 配置日志
logging.basicConfig(
    filename='access_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FaceAccessControl:
    def __init__(self, data_path='face_data', encodings_file='encodings.pkl'):
        # 初始化人脸识别器
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            # 如果cv2.data不可用，尝试使用相对路径或默认路径
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            print(f"警告: 使用备用路径加载级联分类器: {e}")
            
        # 修改人脸识别器初始化逻辑，增加兼容性
        self.face_recognition_available = True
        try:
            # 尝试OpenCV 3.x/4.x的现代版本方法
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            print("成功初始化LBPHFaceRecognizer (现代方法)")
        except AttributeError:
            try:
                # 尝试OpenCV 2.x的旧版本方法
                self.recognizer = cv2.createLBPHFaceRecognizer()
                print("成功初始化LBPHFaceRecognizer (旧版方法1)")
            except AttributeError:
                try:
                    # 尝试另一种旧版本方法
                    self.recognizer = cv2.face.createLBPHFaceRecognizer()
                    print("成功初始化LBPHFaceRecognizer (旧版方法2)")
                except AttributeError as e:
                    print(f"错误: 无法初始化人脸识别器: {e}")
                    print("已创建模拟识别器，人脸识别功能将受限")
                    self.face_recognition_available = False
                    # 创建一个模拟的识别器对象，避免程序崩溃
                    class DummyRecognizer:
                        def read(self, *args):
                            pass
                        def predict(self, *args):
                            return (0, 0.0)
                    self.recognizer = DummyRecognizer()

        # 数据存储路径
        self.data_path = data_path
        self.encodings_file = encodings_file
        self.trainer_file = os.path.join(data_path, 'trainer.yml')
        self.users_file = os.path.join(data_path, 'users.pkl')

        # 确保数据目录存在
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        # 加载已注册用户
        self.users = self._load_users()
        self.last_access_time = {}  # 记录用户上次访问时间，防止短时间内重复开门
        
        # 初始化TTS（如果可用）
        self.TTS_AVAILABLE = False
        self.tts_engine = None
        self._init_tts()

    def _init_tts(self):
        """初始化文本转语音引擎"""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.TTS_AVAILABLE = True
            print("语音提示功能已启用")
        except ImportError:
            print("未安装pyttsx3库，语音提示功能不可用")
            logging.warning("未安装pyttsx3库，语音提示功能不可用")

    def _speak(self, text):
        """语音播报文本"""
        if self.TTS_AVAILABLE and self.tts_engine:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def _load_users(self):
        """加载已注册用户信息"""
        users = {}
        
        # 首先尝试从pickle文件加载
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'rb') as f:
                    users = pickle.load(f)
                print(f"从文件加载用户信息: {len(users)} 位用户")
            except:
                print("用户信息文件损坏，从目录结构重新加载")
        
        # 如果pickle文件不存在或损坏，从目录结构加载
        if not users:
            for user_dir in os.listdir(self.data_path):
                user_path = os.path.join(self.data_path, user_dir)
                if os.path.isdir(user_path) and not user_dir.startswith('.'):
                    # 处理目录名格式: id_name 或 person_id
                    try:
                        if user_dir.startswith('person_'):
                            # 对于person_1格式的目录
                            user_id = int(user_dir.split('_')[1])
                            user_name = f"Person_{user_id}"
                        else:
                            # 对于id_name格式的目录
                            parts = user_dir.split('_', 1)
                            user_id = int(parts[0])
                            user_name = parts[1] if len(parts) > 1 else f"User_{user_id}"
                        users[user_id] = user_name
                    except ValueError:
                        print(f"跳过无效的用户目录: {user_dir}")
                        continue
            
            # 保存加载的用户信息
            self._save_users(users)
        
        # 尝试加载训练好的模型
        try:
            if os.path.exists(self.trainer_file):
                self.recognizer.read(self.trainer_file)
                print(f"成功加载模型，已注册用户: {len(users)}")
            else:
                print("未找到已训练的模型，将使用新模型")
        except Exception as e:
            print(f"加载模型失败: {e}")
            logging.error(f"加载模型失败: {e}")

        return users
    
    def _save_users(self, users=None):
        """保存用户信息到文件"""
        if users is None:
            users = self.users
        try:
            with open(self.users_file, 'wb') as f:
                pickle.dump(users, f)
        except Exception as e:
            print(f"保存用户信息失败: {e}")
            logging.error(f"保存用户信息失败: {e}")

    def register_user(self, user_name, sample_count=30):
        """注册新用户"""
        # 生成新用户ID
        if self.users:
            user_id = max(self.users.keys()) + 1
        else:
            user_id = 1

        # 创建用户目录
        user_dir = os.path.join(self.data_path, f"{user_id}_{user_name}")
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        else:
            print(f"用户 {user_name} 已存在，将更新样本")

        # 打开摄像头采集人脸样本
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            logging.error("无法打开摄像头")
            return None, None

        print(f"正在注册 {user_name}，请正视摄像头...")
        print(f"将采集 {sample_count} 个样本，按 'q' 退出")
        
        if self.TTS_AVAILABLE:
            self._speak(f"正在注册{user_name}，请正视摄像头")

        sample_num = 0
        while sample_num < sample_count:
            ret, frame = cap.read()
            if not ret:
                print("无法获取图像帧")
                break

            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 检测人脸
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # 确保人脸区域足够大
                if w > 100 and h > 100:
                    # 绘制人脸矩形
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sample_num += 1
                    # 保存人脸样本
                    face_img = gray[y:y + h, x:x + w]
                    # 调整图像大小以统一尺寸
                    face_img = cv2.resize(face_img, (200, 200))
                    cv2.imwrite(os.path.join(user_dir, f"face_{sample_num}.jpg"), face_img)
                    cv2.putText(frame, f"Sample {sample_num}/{sample_count}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # 显示当前状态
            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Register Face', frame)

            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 稍微延迟一下，避免连续拍摄相同姿态
            time.sleep(0.05)  # 从0.1秒降低到0.05秒

        cap.release()
        cv2.destroyWindow('Register Face')

        # 更新用户列表
        self.users[user_id] = user_name
        self._save_users()
        print(f"注册完成，已采集 {sample_num} 个样本")
        
        if self.TTS_AVAILABLE:
            self._speak(f"注册完成，已采集{sample_num}个样本")

        # 重新训练模型
        self._train_model()

        # 记录日志
        logging.info(f"用户注册: ID={user_id}, 姓名={user_name}, 样本数={sample_num}")

        return user_id, user_name

    def delete_user(self, user_id):
        """删除用户"""
        if user_id not in self.users:
            print(f"用户ID {user_id} 不存在")
            return False

        user_name = self.users[user_id]
        user_dir = os.path.join(self.data_path, f"{user_id}_{user_name}")
        
        # 删除用户目录和文件
        if os.path.exists(user_dir):
            import shutil
            try:
                shutil.rmtree(user_dir)
                print(f"已删除用户 {user_name} 的样本文件")
            except Exception as e:
                print(f"删除用户文件失败: {e}")
                logging.error(f"删除用户文件失败: {e}")
                return False
        
        # 从用户列表中移除
        del self.users[user_id]
        self._save_users()
        
        # 如果是最后一个用户，删除模型文件
        if not self.users and os.path.exists(self.trainer_file):
            try:
                os.remove(self.trainer_file)
                print("已删除模型文件")
            except Exception as e:
                print(f"删除模型文件失败: {e}")
                logging.error(f"删除模型文件失败: {e}")
        else:
            # 重新训练模型
            self._train_model()
        
        print(f"用户 {user_name} (ID: {user_id}) 已删除")
        logging.info(f"用户删除: ID={user_id}, 姓名={user_name}")
        
        return True

    def _train_model(self):
        """训练人脸识别模型"""
        faces = []
        ids = []

        # 收集所有样本
        for user_id, user_name in self.users.items():
            user_dir = os.path.join(self.data_path, f"{user_id}_{user_name}")
            if not os.path.exists(user_dir):
                continue
            
            for img_file in os.listdir(user_dir):
                img_path = os.path.join(user_dir, img_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # 确保图像尺寸一致
                    img = cv2.resize(img, (200, 200))
                    faces.append(img)
                    ids.append(user_id)
                except Exception as e:
                    print(f"加载图像 {img_file} 失败: {e}")
                    logging.error(f"加载图像 {img_file} 失败: {e}")
                    continue

        if faces and ids:
            # 训练模型
            try:
                self.recognizer.train(faces, np.array(ids))
                # 保存模型
                self.recognizer.write(self.trainer_file)
                print(f"模型训练完成，使用 {len(faces)} 个样本")
                logging.info(f"模型训练完成，使用 {len(faces)} 个样本，{len(set(ids))} 位用户")
            except Exception as e:
                print(f"模型训练失败: {e}")
                logging.error(f"模型训练失败: {e}")
        else:
            print("没有足够的样本进行训练")
            logging.warning("没有足够的样本进行训练")

    def recognize_face(self, threshold=80, door_open_time=5):
        """识别人脸并控制门禁"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            logging.error("无法打开摄像头")
            return

        print("开始人脸识别，按 'q' 退出")
        if self.TTS_AVAILABLE:
            self._speak("开始人脸识别")

        # 创建窗口并设置名称
        cv2.namedWindow('Face Recognition Access Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Recognition Access Control', 800, 600)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取图像帧")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # 绘制状态信息背景
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, frame.shape[0]-30), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)

            for (x, y, w, h) in faces:
                # 识别人脸
                id_, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])

                # 计算置信度百分比 (0 表示完美匹配)
                confidence_percent = 100 - confidence

                # 判断是否识别成功
                if confidence_percent >= (100 - threshold):
                    user_name = self.users.get(id_, "Unknown")
                    color = (0, 255, 0)  # 绿色表示识别成功
                    text = f"{user_name} ({confidence_percent:.1f}%)"

                    # 控制门禁
                    current_time = time.time()
                    if (user_name not in self.last_access_time or
                            current_time - self.last_access_time[user_name] > door_open_time):
                        self._open_door(user_name)
                        self.last_access_time[user_name] = current_time
                else:
                    text = f"Unknown ({confidence_percent:.1f}%)"
                    color = (0, 0, 255)  # 红色表示识别失败

                # 绘制矩形和文本
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # 显示系统状态和时间
            cv2.putText(frame, "Face Recognition Access Control System",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Registered Users: {len(self.users)}",
                        (frame.shape[1] - 300, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Face Recognition Access Control', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        if self.TTS_AVAILABLE:
            self._speak("人脸识别已结束")

    def _open_door(self, user_name):
        """模拟开门操作"""
        print(f"\n[ACCESS GRANTED] 欢迎 {user_name}！")
        print(f"[{datetime.now()}] 门已打开，将在5秒后关闭")
        logging.info(f"门禁开启: 用户={user_name}")

        if self.TTS_AVAILABLE:
            self._speak(f"欢迎{user_name}")

        # 在实际应用中，这里会连接硬件控制门禁
        # 例如：控制继电器打开门锁
        # import RPi.GPIO as GPIO
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setup(relay_pin, GPIO.OUT)
        # GPIO.output(relay_pin, GPIO.HIGH)  # 打开继电器
        # time.sleep(5)  # 保持5秒
        # GPIO.output(relay_pin, GPIO.LOW)   # 关闭继电器

    def list_users(self):
        """列出所有已注册用户"""
        if not self.users:
            print("\n暂无已注册用户")
            return
        
        print("\n=== 已注册用户列表 ===")
        print("{:<5} {:<20} {:<10}".format("ID", "姓名", "样本数"))
        print("=" * 40)
        
        for user_id, user_name in sorted(self.users.items()):
            # 计算用户样本数量
            user_dir = os.path.join(self.data_path, f"{user_id}_{user_name}")
            sample_count = 0
            if os.path.exists(user_dir):
                sample_count = len([f for f in os.listdir(user_dir) if f.endswith(('.jpg', '.png'))])
            
            print("{:<5} {:<20} {:<10}".format(user_id, user_name, sample_count))
        
        print("=" * 40)
        print(f"总计 {len(self.users)} 位用户")

    def export_logs(self, output_file='access_log_export.txt'):
        """导出访问日志"""
        if not os.path.exists('access_log.log'):
            print("没有找到访问日志文件")
            return False
        
        try:
            with open('access_log.log', 'r') as f:
                logs = f.readlines()
            
            with open(output_file, 'w') as f:
                f.write("==== 访问日志导出 ====\n")
                f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总记录数: {len(logs)}\n\n")
                f.writelines(logs)
            
            print(f"日志已导出到 {output_file}")
            logging.info(f"日志导出成功，文件: {output_file}")
            return True
        except Exception as e:
            print(f"导出日志失败: {e}")
            logging.error(f"导出日志失败: {e}")
            return False


def main():
    # 初始化系统
    try:
        access_control = FaceAccessControl()
        
        while True:
            print("\n" + "=" * 30)
            print("   人脸识别门禁系统")
            print("=" * 30)
            print("1. 注册新用户")
            print("2. 开始人脸识别")
            print("3. 查看已注册用户")
            print("4. 删除用户")
            print("5. 导出访问日志")
            print("0. 退出系统")
            print("=" * 30)

            choice = input("请选择操作 (0-5): ")

            if choice == '1':
                user_name = input("请输入用户名: ")
                access_control.register_user(user_name)
            elif choice == '2':
                try:
                    threshold = float(input("请输入识别阈值 (默认80，值越小越严格): ") or "80")
                    door_time = float(input("请输入门开启时间 (默认5秒): ") or "5")
                    access_control.recognize_face(threshold, door_time)
                except ValueError:
                    print("输入无效，使用默认值")
                    access_control.recognize_face()
            elif choice == '3':
                access_control.list_users()
            elif choice == '4':
                access_control.list_users()
                try:
                    user_id = int(input("请输入要删除的用户ID: "))
                    access_control.delete_user(user_id)
                except ValueError:
                    print("无效的用户ID")
            elif choice == '5':
                output_file = input("请输入导出文件名 (默认: access_log_export.txt): ") or "access_log_export.txt"
                access_control.export_logs(output_file)
            elif choice == '0':
                print("系统已退出")
                break
            else:
                print("无效的选择，请重试")
    except Exception as e:
        print(f"系统发生错误: {e}")
        logging.error(f"系统错误: {e}")


if __name__ == "__main__":
    main()