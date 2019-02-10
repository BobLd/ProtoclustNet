using System;

namespace ProtoclustNet
{
    /*
     * Bien, J., and Tibshirani, R. (2011), "Hierarchical Clustering
     * with Prototypes via Minimax Linkage," \emph{The Journal of the American
     * Statistical Association}, 106(495), 1075-1084.
     * 
     * Murtagh, F. (1983), "A Survey of Recent Advances in Hierarchical Clustering
     * Algorithms," \emph{The Computer Journal}, \bold{26}, 354--359.
     */

    /// <summary>
    /// TO DO
    /// Cut a Minimax Linkage Tree To Get a Clustering
    /// </summary>
    public static class Protocut
    {
        /// <summary>
        /// Cut a Minimax Linkage Tree To Get a Clustering
        /// <para>Cuts a minimax linkage tree to get one of n - 1 clusterings.</para>
        /// </summary>
        /// <param name="merge"></param>
        /// <param name="height"></param>
        /// <param name="order"></param>
        /// <param name="protos"></param>
        /// <param name="k"></param>
        public static void Compute(int[][] merge, double[] height, int[] order, int[] protos, int k)
        {
            /*
             * https://github.com/jacobbien/protoclust/blob/master/R/protocut.R
             */

            int n = height.Length + 1;
            throw new NotImplementedException("Protocut");
        }
    }
}
