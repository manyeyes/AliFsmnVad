using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVadSharp.DLL
{
    internal struct FbankData
    {
        //[MarshalAs(UnmanagedType.ByValArray, SizeConst = 80)]
        //public float[] data;
        public IntPtr data;
        public int data_length;
    };

    internal struct FbankDatas
    {
        public IntPtr data;
        public int data_length;
    };

    internal struct FbankData300000
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 300000)]
        public float[] data;
        public int data_length;
    };

    

    internal struct KnfOnlineFbank
    {
        public IntPtr impl;
    };
}
