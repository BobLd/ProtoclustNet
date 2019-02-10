using System;
using System.IO;
using System.Linq;

namespace ProtoclustNet.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = @"distance_small.csv";
            var lines = File.ReadAllLines(path);
            var values = lines.Skip(1).Select(line => line.Split(',').Select(x => double.Parse(x)).ToArray()).ToArray();

            Protoclust.Compute(values, out int[][] merge, out double[] height, out int[] order, out int[] protos);

            Console.WriteLine("merge:");
            Console.WriteLine(ToStringInt(merge));

            Console.WriteLine("height:");
            Console.WriteLine(ToStringDbl(height));

            Console.WriteLine("order:");
            Console.WriteLine(ToStringInt(order));

            Console.WriteLine("protos:");
            Console.WriteLine(ToStringInt(protos));
            Console.ReadKey();
        }

        public static string ToStringDbl(double[] arr)
        {
            string ret = "";
            for (int i = 0; i < arr.Length; i++)
            {
                ret = ret + "[" + i + "]\t" + arr[i] + "\n";
            }
            return ret;
        }

        public static string ToStringInt(int[] arr)
        {
            string ret = "";
            for (int i = 0; i < arr.Length; i++)
            {
                ret = ret + "[" + i + "]\t" + arr[i] + "\n";
            }
            return ret;
        }

        public static string ToStringInt(int[][] arr)
        {
            string ret = "";
            for (int i = 0; i < arr[0].Length; i++)
            {
                ret = ret + "[" + i + "]\t" + arr[0][i] + "\t" + arr[1][i] + "\n";
            }
            return ret;
        }
    }
}
