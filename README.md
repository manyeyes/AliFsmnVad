# AliFsmnVad
##### 简介：
项目中使用的VAD模型是阿里巴巴达摩院提供的FSMN-Monophone VAD模型。
**项目基于Net 6.0，使用C#编写，调用Microsoft.ML.OnnxRuntime对onnx模型进行解码，支持跨平台编译。项目以库的形式进行调用，部署非常方便。**
VAD整体流程的rtf在0.008左右。

##### 用途：
16k中文通用VAD模型：可用于检测长语音片段中有效语音的起止时间点.
FSMN-Monophone VAD是达摩院语音团队提出的高效语音端点检测模型，用于检测输入音频中有效语音的起止时间点信息，并将检测出来的有效音频片段输入识别引擎进行识别，减少无效语音带来的识别错误。

##### VAD常用参数调整说明（参考：vad.yaml文件）：
max_end_silence_time：尾部连续检测到多长时间静音进行尾点判停，参数范围500ms～6000ms，默认值800ms(该值过低容易出现语音提前截断的情况)。
speech_noise_thres：speech的得分减去noise的得分大于此值则判断为speech，参数范围：（-1,1）
取值越趋于-1，噪音被误判定为语音的概率越大，FA越高
取值越趋于+1，语音被误判定为噪音的概率越大，Pmiss越高
通常情况下，该值会根据当前模型在长语音测试集上的效果取balance

##### 调用方式：
###### 1.添加项目引用
```csharp
using AliFsmnVad;
using AliFsmnVad.Model;
using NAudio.Wave;
```

###### 2.初始化模型和配置
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "speech_fsmn_vad_zh-cn-16k-common-onnx";
string modelFilePath = applicationBase + "./"+ modelName + "/model.onnx";
string configFilePath = applicationBase + "./"+ modelName + "/vad.yaml";
string mvnFilePath = applicationBase + "./"+ modelName + "/vad.mvn";
int batchSize = 2;//批量解码
AliFsmnVad aliFsmnVad = new AliFsmnVad(modelFilePath, configFilePath, mvnFilePath, batchSize);
```
###### 3.调用
方法一(适用于小文件)：
```csharp
SegmentEntity[] segments_duration = aliFsmnVad.GetSegments(samples);
```
方法二(适用于大文件)：
```csharp
SegmentEntity[] segments_duration = aliFsmnVad.GetSegmentsByStep(samples);
```
###### 4.获取结果
```csharp
//遍历segments_duration
foreach (SegmentEntity segment in segments_duration)
{
    //segment.Waveform 是vad切分后的语音片段sample
	//segment.Segment 是每一段语音对应的时间戳
}
```
输出相应时间戳:
```
load model and init config elapsed_milliseconds:463.5390625
vad infer result:
[[70,2340][2620,6200][6480,23670][23950,26250][26780,28990][29950,31430][31750,37600][38210,46900][47310,49630][49910,56460][56740,59540][59820,70450]]
elapsed_milliseconds:662.796875
total_duration:70470.625
rtf:0.009405292985552491
```
输出的数据，例如：[70,2340]，是以毫秒为单位的segement的起止时间，可以以此为依据对音频进行分片。其中静音噪音部分已被去除。

##### 语音识别
将segment.Waveform（见上述4）作为参数，调用 AliParaformerAsr 、K2TransducerAsr、SherpaOnnxSharp的offlineRecognizer的相应方法执行语音识别任务。具体请参考相应示例。

##### 其他说明：
测试用例：AliFsmnVad.Examples。
支持平台：
Windows 7 SP1或更高版本,
macOS 10.13 (High Sierra) 或更高版本,ios等，
Linux 发行版（需要特定的依赖关系，详见.NET 6支持的Linux发行版列表），
Android（Android 5.0 (API 21) 或更高版本）。

测试用例中samples的计算,使用的是NAudio库。

##### 模型下载
https://huggingface.co/manyeyes/speech_fsmn_vad_zh-cn-16k-common-onnx
https://www.modelscope.cn/models/manyeyes/alifsmnvad-onnx

通过以下链接了解更多：
https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx/summary
