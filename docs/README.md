# EmotiCompanion User Study

这个工具包是一个**单文件、离线、无需服务器**的交互式 HTML。下载并双击 `emoticompanion_user_study.html` 即可在浏览器中使用。

## 包含内容

- 实验前材料清单与研究者设置
- 知情同意书与可打印版本
- 基线情绪/疲劳/音乐需求自评
- 情绪诱导 manipulation check
- 系统感知与融合结果记录
- Music A / Music B 盲评
- A/B 偏好与揭盲映射
- 体验后状态、隐私与等待时间问卷
- 开放访谈记录
- 本地自动保存、参与者记录管理
- 单人 CSV/JSON 导出与全部参与者 CSV 汇总

## 推荐操作流程

1. 使用 Chrome / Edge 打开 HTML。
2. 研究者先完成“研究设置”，包括 PID、协议、目标状态、A/B 条件顺序与 seed。
3. 切换为“参与者视图”，隐藏 researcher-only 字段。
4. 按左侧步骤完成知情同意、基线、诱导检查、感知和 A/B 评分。
5. 评分结束后切回“研究者视图”，填写 A/B 对应的 ToM / Standard、系统输出与揭盲结果。
6. 在最后一页点击“保存/更新本参与者”。
7. 每位参与者结束后导出单人文件；全部完成后导出 `emoticompanion_user_study_all.csv`。

## 数据与隐私

- 表单数据仅保存在当前浏览器的 `localStorage`，不会上传网络。
- 更换浏览器、清理浏览器数据或无痕模式关闭后，数据可能丢失。
- 每完成一位参与者，请立即导出 CSV/JSON 并离线备份。
- 本工具不读取或保存摄像头、麦克风媒体；媒体保存仍由 `app.py` 的 study setting 控制。

## A/B 盲评原则

- 参与者只看到“音乐 A / 音乐 B”。
- A/B 评分前不要说明哪段是 ToM 或 Standard。
- 尽量冻结同一个 fusion state，只改变 reasoning mode。
- 两段音乐建议使用相同 MusicGen seed。
- 条件顺序交替平衡，例如 P01: A=ToM，P02: A=Standard。
