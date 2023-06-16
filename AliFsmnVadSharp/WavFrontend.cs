using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AliFsmnVadSharp.Model;
using AliFsmnVadSharp.DLL;
using AliFsmnVadSharp.Struct;
using System.Runtime.InteropServices;

namespace AliFsmnVadSharp
{
    internal class WavFrontend
    {
        private string _mvnFilePath;
        private FrontendConfEntity _frontendConfEntity;
        IntPtr _opts = IntPtr.Zero;
        private CmvnEntity _cmvnEntity;

        private static int _fbank_beg_idx = 0;

        public WavFrontend(string mvnFilePath, FrontendConfEntity frontendConfEntity)
        {
            _mvnFilePath = mvnFilePath;
            _frontendConfEntity = frontendConfEntity;
            _fbank_beg_idx = 0;
            _opts = KaldiNativeFbank.GetFbankOptions(
                dither: _frontendConfEntity.dither,
                snip_edges: true,
                sample_rate: _frontendConfEntity.fs,
                num_bins: _frontendConfEntity.n_mels
                );
            _cmvnEntity = LoadCmvn(mvnFilePath);
        }

        public float[] GetFbank(float[] samples)
        {
            float sample_rate = _frontendConfEntity.fs;
            samples = samples.Select((float x) => x * 32768f).ToArray();
            //DLL.FbankDatas fbankDatas = new DLL.FbankDatas();
            // method1
            //FbankDatas fbankDatas = new FbankDatas();
            //KaldiNativeFbank.GetFbanks(_knfOnlineFbank, framesNum,ref fbankDatas);
            // method2
            KnfOnlineFbank _knfOnlineFbank = KaldiNativeFbank.GetOnlineFbank(_opts);
            KaldiNativeFbank.AcceptWaveform(_knfOnlineFbank, sample_rate, samples, samples.Length);
            KaldiNativeFbank.InputFinished(_knfOnlineFbank);
            int framesNum = KaldiNativeFbank.GetNumFramesReady(_knfOnlineFbank);
            float[] fbanks = new float[framesNum * 80];
            for (int i = 0; i < framesNum; i++)
            {
                FbankData fbankData = new FbankData();
                KaldiNativeFbank.GetFbank(_knfOnlineFbank, i, ref fbankData);
                float[] _fbankData = new float[fbankData.data_length];
                Marshal.Copy(fbankData.data, _fbankData, 0, fbankData.data_length);
                Array.Copy(_fbankData, 0, fbanks, i * 80, _fbankData.Length);
                fbankData.data = IntPtr.Zero;
                _fbankData = null;
            }

            samples = null;
            GC.Collect();
            return fbanks;
        }


        public float[] LfrCmvn(float[] fbanks)
        {
            float[] features = fbanks;
            if (_frontendConfEntity.lfr_m != 1 || _frontendConfEntity.lfr_n != 1)
            {
                features = ApplyLfr(fbanks, _frontendConfEntity.lfr_m, _frontendConfEntity.lfr_n);
            }
            if (_cmvnEntity != null)
            {
                features = ApplyCmvn(features);
            }
            return features;
        }

        private float[] ApplyCmvn(float[] inputs)
        {
            var arr_neg_mean = _cmvnEntity.Means;
            float[] neg_mean = arr_neg_mean.Select(x => (float)Convert.ToDouble(x)).ToArray();
            var arr_inv_stddev = _cmvnEntity.Vars;
            float[] inv_stddev = arr_inv_stddev.Select(x => (float)Convert.ToDouble(x)).ToArray();

            int dim = neg_mean.Length;
            int num_frames = inputs.Length / dim;

            for (int i = 0; i < num_frames; i++)
            {
                for (int k = 0; k != dim; ++k)
                {
                    inputs[dim * i + k] = (inputs[dim * i + k] + neg_mean[k]) * inv_stddev[k];
                }
            }
            return inputs;
        }

        public float[] ApplyLfr(float[] inputs, int lfr_m, int lfr_n)
        {            
            //List<float[]> LFR_inputs = new List<float[]>();
            int t = inputs.Length / 80;
            int t_lfr = (int)Math.Floor((double)(t / lfr_n));
            float[] input_0 = new float[80];
            Array.Copy(inputs, 0, input_0, 0, 80);
            int tile_x = (lfr_m - 1) / 2;
            t = t + tile_x;
            float[] inputs_temp = new float[t * 80];
            for (int i = 0; i < tile_x; i++)
            {
                Array.Copy(input_0, 0, inputs_temp, tile_x * 80, 80);
            }
            Array.Copy(inputs, 0, inputs_temp, tile_x * 80, inputs.Length);
            inputs = inputs_temp;

            float[] LFR_outputs = new float[t_lfr * lfr_m * 80];
            for (int i = 0; i < t_lfr; i++)
            {
                if (lfr_m <= t - i * lfr_n)
                {
                    //float[] input_i = new float[lfr_m * 80];
                    //Array.Copy(inputs, i * lfr_n * 80, input_i, 0, input_i.Length);
                    //LFR_inputs.Add(input_i);
                    Array.Copy(inputs, i * lfr_n * 80, LFR_outputs, i* lfr_m * 80, lfr_m * 80);
                }
                else
                {
                    // process last LFR frame
                    int num_padding = lfr_m - (t - i * lfr_n);
                    float[] frame = new float[lfr_m * 80];
                    Array.Copy(inputs, i * lfr_n * 80, frame, 0, (t - i * lfr_n) * 80);//frame.Length

                    for (int j = 0; j < num_padding; j++)
                    {
                        Array.Copy(inputs, (t - 1) * 80, frame, (lfr_m - num_padding + j) * 80, 80);
                    }
                    //LFR_inputs.Add(frame);
                    Array.Copy(frame, 0, LFR_outputs, i * lfr_m * 80, frame.Length);
                }
            }
            return LFR_outputs;
        }

        public float[] ApplyLfr_originalVersion(float[] inputs, int lfr_m, int lfr_n)
        {
            List<float[]> LFR_inputs = new List<float[]>();

            int t = inputs.Length / 80;
            int t_lfr = (int)Math.Floor((double)(t / lfr_n));
            float[] input_0 = new float[80];
            Array.Copy(inputs, 0, input_0, 0, 80);
            int tile_x = (lfr_m - 1) / 2;
            //for(int i = 0; i < tile_x; i++)
            //{
            //    LFR_inputs.Add(input_0);
            //}
            //float[] left_padding = np.tile(inputs[0], ((lfr_m - 1) / 2, 1));
            //float[] inputs = np.vstack((left_padding, inputs));
            t = t + tile_x;
            float[] inputs_temp = new float[t * 80];
            for (int i = 0; i < tile_x; i++)
            {
                Array.Copy(input_0, 0, inputs_temp, tile_x * 80, 80);
            }
            //Array.Copy(input_0, 0, inputs_temp, 0, 80);
            //Array.Copy(input_0, 0, inputs_temp, 80, 80);
            Array.Copy(inputs, 0, inputs_temp, tile_x * 80, inputs.Length);
            inputs = inputs_temp;
            for (int i = 0; i < t_lfr; i++)
            {
                if (lfr_m <= t - i * lfr_n)
                {
                    float[] input_i = new float[lfr_m * 80];
                    Array.Copy(inputs, i * lfr_n * 80, input_i, 0, input_i.Length);
                    LFR_inputs.Add(input_i);
                }
                else
                {
                    // process last LFR frame
                    int num_padding = lfr_m - (t - i * lfr_n);
                    float[] frame = new float[lfr_m * 80];
                    Array.Copy(inputs, i * lfr_n * 80, frame, 0, (t - i * lfr_n) * 80);//frame.Length

                    for (int j = 0; j < num_padding; j++)
                    {
                        //frame = np.hstack((frame, inputs[-1]));
                        Array.Copy(inputs, (t - 1) * 80, frame, (lfr_m - num_padding + j) * 80, 80);
                    }
                    LFR_inputs.Add(frame);
                    //float[] frame = inputs[i * lfr_n:].reshape(-1);
                    //for (int j = 0; j < num_padding; j++)
                    //{
                    //    frame = np.hstack((frame, inputs[-1]));
                    //}
                    //LFR_inputs.append(frame);

                }
            }
            //////////////////////////
            inputs = null;
            float[] LFR_outputs = new float[t_lfr * lfr_m * 80];
            int p = 0;
            foreach (float[] f in LFR_inputs)
            {
                Array.Copy(f, 0, LFR_outputs, p * lfr_m * 80, f.Length);
                p++;
            }
            return LFR_outputs;
        }
        
        private CmvnEntity LoadCmvn(string mvnFilePath)// -> np.ndarray:
        {
            List<float> means_list = new List<float>();
            List<float> vars_list = new List<float>();
            FileStreamOptions options = new FileStreamOptions();
            options.Access = FileAccess.Read;
            options.Mode = FileMode.Open;
            StreamReader srtReader = new StreamReader(mvnFilePath, options);
            int i = 0;
            while (!srtReader.EndOfStream)
            {
                string? strLine = srtReader.ReadLine();
                if (!string.IsNullOrEmpty(strLine))
                {
                    if (strLine.StartsWith("<AddShift>")) 
                    {
                        i=1;
                        continue;
                    }
                    if (strLine.StartsWith("<Rescale>"))
                    {
                        i = 2;
                        continue;
                    }
                    if (strLine.StartsWith("<LearnRateCoef>") && i==1)
                    {
                        string[] add_shift_line = strLine.Substring(strLine.IndexOf("[") + 1, strLine.LastIndexOf("]") - strLine.IndexOf("[") - 1).Split(" ");
                        means_list = add_shift_line.Where(x => !string.IsNullOrEmpty(x)).Select(x => float.Parse(x.Trim())).ToList();
                        //i++;
                        continue;
                    }
                    if (strLine.StartsWith("<LearnRateCoef>") && i==2)
                    {
                        string[] rescale_line = strLine.Substring(strLine.IndexOf("[") + 1, strLine.LastIndexOf("]") - strLine.IndexOf("[") - 1).Split(" ");
                        vars_list = rescale_line.Where(x => !string.IsNullOrEmpty(x)).Select(x => float.Parse(x.Trim())).ToList();
                        //i++;
                        continue;
                    }
                }
            }
            CmvnEntity cmvnEntity = new CmvnEntity();
            cmvnEntity.Means = means_list;
            cmvnEntity.Vars = vars_list;
            return cmvnEntity;
        }

    }
}
