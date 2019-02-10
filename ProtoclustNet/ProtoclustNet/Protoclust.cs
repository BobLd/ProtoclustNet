using System;

namespace ProtoclustNet
{
    /// <summary>
    /// Minimax linkage agglomerative clustering.
    /// <para>https://github.com/jacobbien/protoclust</para>
    /// </summary>
    public static class Protoclust
    {
        /// <summary>
        /// Minimax linkage agglomerative clustering.
        /// <para>https://github.com/jacobbien/protoclust</para>
        /// </summary>
        /// <param name="d">The distance vector</param>
        /// <param name="merge"></param>
        /// <param name="height"></param>
        /// <param name="order"></param>
        /// <param name="protos"></param>
        public static void Compute(double[] d, out int[][] merge, out double[] height, out int[] order, out int[] protos)
        {
            int nn = d.Length;
            int n = (int)(1 + Math.Sqrt(1 + 8 * nn)) / 2;

            int[] mergeVect = new int[(n - 1) * 2];
            height = new double[n - 1];
            order = new int[n];
            protos = new int[n - 1];

            hier(d, n, 0, mergeVect, height, order, protos);

            merge = new int[2][];

            merge[0] = new int[n - 1];
            merge[1] = new int[n - 1];

            for (int r = 0; r < n - 1; r++)
            {
                merge[0][r] = mergeVect[2 * r];
                merge[1][r] = mergeVect[2 * r + 1];
            }
        }

        /// <summary>
        /// Minimax linkage agglomerative clustering.
        /// <para>https://github.com/jacobbien/protoclust</para>
        /// </summary>
        /// <param name="d">The distance matrix</param>
        /// <param name="merge"></param>
        /// <param name="height"></param>
        /// <param name="order"></param>
        /// <param name="protos"></param>
        public static void Compute(double[][] d, out int[][] merge, out double[] height, out int[] order, out int[] protos)
        {
            Compute(ToDist(d), out merge, out height, out order, out protos);
        }

        /// <summary>
        /// Minimax linkage agglomerative clustering.
        /// <para>https://github.com/jacobbien/protoclust</para>
        /// </summary>
        /// <param name="d"></param>
        /// <param name="ndim"></param>
        /// <param name="verbose"></param>
        /// <param name="merge"></param>
        /// <param name="height"></param>
        /// <param name="order"></param>
        /// <param name="protos"></param>
        public static void hier(double[] d, int ndim, int verbose, int[] merge, double[] height, int[] order, int[] protos)
        {
            int i = 0;
            int j = 0;
            int k = 0;
            int ii = 0;
            int imerge = 0;
            int jmerge = 0;
            int reverse = 0;

            int n = ndim;

            double[] dd = new double[(n * (n - 1) / 2)];
            // dd contains the inter-cluster distances as a lower triangular matrix

            double[] dmax = new double[n * n];
            // dmax[n*i + j] = max_{l\in C_j}d(i,l) where C_j is the jth cluster

            int[] un_protos = new int[n];
            int[] un_merge = new int[n * 2];

            // versions of the outputs "protos" and "merge"... except unordered by height
            // these will be ordered based on the nearest-neighbor chain order
            // at the end, we will convert from un_protos to protos and likewise for merge

            // each cluster will be stored as a linked list between clusters[i] and tails[i]
            Cluster[] clusters = new Cluster[n];
            Cluster[] tails = new Cluster[n];

            var clustLabel = new int[n];
            var clustSize = new int[n];

            var nnchain = new int[n];
            int nn, end = -1;           // index of end of nnchain
            var inchain = new int[n];   // indicates whether element i is in current chain

            double old;
            int nochange = 0;

            // initialize n singleton clusters
            for (i = 0; i < n; i++)
            {
                clusters[i] = new Cluster();
                tails[i] = clusters[i];
                clusters[i].i = i;
                clusters[i].next = null;
                clustLabel[i] = -(i + 1);// leaves are negative (as in R's hclust)
                clustSize[i] = 1;
                nnchain[i] = -1;
                inchain[i] = 0;
            }
            for (i = 0; i < n; i++)
            {
                dmax[n * i + i] = 0;
            }

            for (j = 0; j < n - 1; j++)
            {
                for (i = j + 1; i < n; i++)
                {
                    ii = lt(i, j, n);
                    dmax[n * i + j] = d[ii];
                    dmax[n * j + i] = d[ii];
                }
            }

            for (ii = 0; ii < n * (n - 1) / 2; ii++)
            {
                dd[ii] = d[ii];
            }

            for (k = 0; k < n - 1; k++)
            {
                // check if chain is empty
                if (end == -1)
                {
                    // start a new chain
                    for (i = 0; i < n; i++)
                    {
                        if (clustLabel[i] != 0) break;
                    }
                    nnchain[0] = i;
                    end = 0;
                }

                // grow nearest neighbor chain until a RNN pair found:
                while (true)
                {
                    nn = findNN(dd, n, clustLabel, dmax, clusters, inchain, nnchain[end]);
                    if (end > 0)
                    {
                        if (nn == nnchain[end - 1]) break; // reached a RNN pair
                        inchain[nnchain[end - 1]] = 1;
                        // we purposely delay this update to allow findNN to 
                        // select a cluster visited a step earlier
                        // (which happens for RNN pairs)
                    }
                    nnchain[++end] = nn;
                }

                if (nnchain[end] < nnchain[end - 1])
                {
                    imerge = nnchain[end - 1];
                    jmerge = nnchain[end];
                }
                else
                {
                    imerge = nnchain[end];
                    jmerge = nnchain[end - 1];
                }

                // remove RNN pair from chain
                inchain[nnchain[end]] = 0;
                inchain[nnchain[end - 1]] = 0;

                if (end > 1) inchain[nnchain[end - 2]] = 0; // again, so that a RNN can be detected

                if (end > 2) inchain[nnchain[end - 3]] = 0; // again, so that a RNN can be detected

                nnchain[end] = -1;
                nnchain[end - 1] = -1;
                end -= 2;

                // create merged cluster from this RNN pair:
                ii = lt(imerge, jmerge, n);
                height[k] = dd[ii];

                reverse = 0;

                if (clustLabel[imerge] > clustLabel[jmerge]) reverse = 1; // put smaller cluster on left                

                if (clustLabel[imerge] < 0 && clustLabel[jmerge] < 0) reverse = 1;  // since imerge > jmerge

                if (reverse == 1)
                {
                    un_merge[2 * k] = clustLabel[jmerge];
                    un_merge[2 * k + 1] = clustLabel[imerge];
                }
                else
                {
                    un_merge[2 * k] = clustLabel[imerge];
                    un_merge[2 * k + 1] = clustLabel[jmerge];
                }

                // update the imerge cluster:
                clustLabel[imerge] = k + 1;
                clustSize[imerge] += clustSize[jmerge];

                if (reverse == 1)
                {
                    // put jmerge's elements to left of imerge's
                    tails[jmerge].next = clusters[imerge];
                    clusters[imerge] = clusters[jmerge];
                }
                else
                {
                    // put imerge's elements to left of jmerge's
                    tails[imerge].next = clusters[jmerge];
                    tails[imerge] = tails[jmerge];
                }

                // at jmerge, we no longer have a cluster:
                clustLabel[jmerge] = 0;
                clustSize[jmerge] = 0;
                clusters[jmerge] = null;
                tails[jmerge] = null;

                /// update dmax:
                for (i = 0; i < n; i++)
                {
                    // point i
                    if (dmax[n * i + imerge] < dmax[n * i + jmerge])
                    {
                        dmax[n * i + imerge] = dmax[n * i + jmerge];//i.e. max{dmax(i,imerge),dmax(i,jmerge)}
                    }
                }

                // get the minimax prototype for this newly formed cluster
                un_protos[k] = minimaxpoint(dmax, n, clusters[imerge], imerge, clustSize[imerge]) + 1;

                /// update dd:

                // imerge is now a new cluster, so update its distances:
                for (j = 0; j < imerge; j++)
                {
                    if (clustLabel[j] != 0)
                    {
                        // still an active cluster
                        ii = lt(imerge, j, n);
                        old = dd[ii];
                        dd[ii] = minimaxlink(dmax, n, clusters[j], tails[j], j, clusters[imerge], imerge, clustSize[j] + clustSize[imerge]);
                        if (dd[ii] == old) nochange++;
                    }
                }

                ii = lt(imerge + 1, imerge, n);
                for (i = imerge + 1; i < n; i++)
                {
                    if (clustLabel[i] != 0)
                    {
                        old = dd[ii];
                        dd[ii] = minimaxlink(dmax, n, clusters[i], tails[i], i, clusters[imerge], imerge, clustSize[i] + clustSize[imerge]);
                        if (dd[ii] == old) nochange++;
                    }
                    ii++;
                }
            }

            // List merges and protos in order of increasing height:
            var o = new int[n - 1];
            for (i = 0; i < n - 1; i++)
            {
                o[i] = i;
            }

            // sort heights and "o = order(height)" (in R speak)
            rsort_with_index(height, o, n - 1);

            // if there are ties, want indices ordered (to match R's convention).
            int count;

            for (i = 0; i < n - 1; i++)
            {
                count = 0;
                if ((i + count + 1) < (n - 1))
                    while (height[i + count + 1] == height[i])
                        count++;
                // heights are constant from i to i+count
                if (count > 0)
                {
                    Console.WriteLine("R_isort(&o[i], count + 1);");
                    //R_isort(&o[i], count + 1);
                }
                i += count;
            }

            for (i = 0; i < n - 1; i++)
            {
                protos[i] = un_protos[o[i]];
                merge[2 * i] = un_merge[2 * o[i]];
                merge[2 * i + 1] = un_merge[2 * o[i] + 1];
            }

            // shuffling merge rows around messes up the positive indices in merge:
            var ranks = new int[n - 1];
            for (i = 0; i < n - 1; i++)
                ranks[o[i]] = i;

            int mtemp;
            for (i = 0; i < n - 1; i++)
            {
                for (j = 0; j < 2; j++)
                {
                    if (merge[2 * i + j] > 0) merge[2 * i + j] = ranks[merge[2 * i + j] - 1] + 1;
                    //the -1 and +1 are to match R's indexing
                }

                if (merge[2 * i] > 0 && merge[2 * i + 1] > 0)
                {
                    if (merge[2 * i] > merge[2 * i + 1])
                    {
                        // hclust has positive rows in increasing order:
                        mtemp = merge[2 * i];
                        merge[2 * i] = merge[2 * i + 1];
                        merge[2 * i + 1] = mtemp;
                    }
                }
            }
            // get order by following "merge"

            // using ranks for a different purpose... ranks[k] will be s.t. clusters[ranks[k]] contains
            // the cluster created at step k.
            for (k = 0; k < n - 1; k++)
            {
                ranks[k] = 0;
            }

            Cluster cur = clusters[imerge];
            for (i = 0; i < n; i++)
            {
                clusters[cur.i] = cur;
                tails[cur.i] = clusters[cur.i];
                cur = cur.next;
            }

            for (i = 0; i < n; i++)
            {
                clusters[i].next = null;
            }

            for (k = 0; k < n - 1; k++)
            {
                if (merge[2 * k] < 0)
                {
                    imerge = -merge[2 * k] - 1;
                }
                else
                {
                    imerge = ranks[merge[2 * k] - 1];
                }

                if (merge[2 * k + 1] < 0)
                {
                    jmerge = -merge[2 * k + 1] - 1;
                }
                else
                {
                    jmerge = ranks[merge[2 * k + 1] - 1];
                }

                tails[imerge].next = clusters[jmerge];
                tails[imerge] = tails[jmerge];
                clusters[jmerge] = null;
                tails[jmerge] = null;
                ranks[k] = imerge;
            }

            cur = clusters[imerge];
            for (i = 0; i < n; i++)
            {
                order[i] = cur.i + 1;
                cur = cur.next;
            }

            //cur = clusters[imerge];
            //Cluster curnext;
            //for (i = 0; i < n; i++)
            //{
            //    curnext = cur.next;
            //    cur = curnext;
            //}
        }

        private static int lt(int i, int j, int n)
        {
            return n * j - j * (j + 1) / 2 + i - j - 1;
        }

        // returns the nearest neighbor cluster of cluster i that is not 
        // already in the chain.
        private static int findNN(double[] dd, int n, int[] clustLabel, double[] dmax, Cluster[] clusters, int[] inchain, int i)
        {
            //int j, ii;
            int ii = 0;
            double mind = double.MaxValue;
            double dcomp, mincomplete = 0;
            int nn = 0;
            for (int j = 0; j < i; j++)
            {
                if (clustLabel[j] == 0 || inchain[j] == 1) continue;
                ii = lt(i, j, n);

                if (dd[ii] < mind)
                {
                    mind = dd[ii];
                    nn = j;
                    mincomplete = 0;// reset mincomplete
                }
                else if (dd[ii] == mind)
                {
                    if (mincomplete == 0)
                    {
                        // this is the first duplicate
                        mincomplete = completelink(dmax, n, clusters[nn], nn, clusters[i], i);
                    }
                    dcomp = completelink(dmax, n, clusters[j], j, clusters[i], i);
                    if (dcomp < mincomplete)
                    {
                        mincomplete = dcomp;
                        nn = j;
                    }
                }
            }

            for (int j = i + 1; j < n; j++)
            {
                if (clustLabel[j] == 0 || inchain[j] == 1) continue;
                ii = lt(j, i, n);
                if (dd[ii] < mind)
                {
                    mind = dd[ii];
                    nn = j;
                    mincomplete = 0;// reset mincomplete
                }
                else if (dd[ii] == mind)
                {
                    if (mincomplete == 0)
                    {
                        // this is the first duplicate
                        mincomplete = completelink(dmax, n, clusters[nn], nn, clusters[i], i);
                    }
                    dcomp = completelink(dmax, n, clusters[j], j, clusters[i], i);

                    if (dcomp < mincomplete)
                    {
                        mincomplete = dcomp;
                        nn = j;
                    }
                }
            }

            return nn;
        }

        /// <summary>
        /// Returns the minimax distance
        /// </summary>
        /// <param name="dmax"></param>
        /// <param name="n"></param>
        /// <param name="G"></param>
        /// <param name="tG"></param>
        /// <param name="iG"></param>
        /// <param name="H"></param>
        /// <param name="iH"></param>
        /// <param name="nGH"></param>
        /// <returns></returns>
        private static double minimaxlink(double[] dmax, int n, Cluster G, Cluster tG, int iG, Cluster H, int iH, int nGH)
        {
            // temporarily combine clusters
            tG.next = H;
            int i;
            double dmm;
            var dmaxGH = new double[nGH];

            Cluster cur1;
            cur1 = G;
            for (i = 0; i < nGH; i++)
            {
                if (dmax[n * cur1.i + iG] > dmax[n * cur1.i + iH])
                {
                    dmaxGH[i] = dmax[n * cur1.i + iG];
                }
                else
                {
                    dmaxGH[i] = dmax[n * cur1.i + iH];
                }
                cur1 = cur1.next;
            }

            dmm = double.MaxValue;
            for (i = 0; i < nGH; i++)
            {
                if (dmaxGH[i] < dmm) dmm = dmaxGH[i];
            }

            // uncombine the clusters
            tG.next = null;
            return dmm;
        }

        /// <summary>
        /// Finds the minimax point of the cluster G
        /// </summary>
        /// <param name="dmax"></param>
        /// <param name="n"></param>
        /// <param name="G"></param>
        /// <param name="iG"></param>
        /// <param name="nG"></param>
        /// <returns></returns>
        private static int minimaxpoint(double[] dmax, int n, Cluster G, int iG, int nG)
        {
            int mm = 0;
            Cluster cur1 = G;
            double dmm = double.MaxValue;

            for (int i = 0; i < nG; i++)
            {
                if (dmax[n * cur1.i + iG] < dmm)
                {
                    dmm = dmax[n * cur1.i + iG];
                    mm = cur1.i;
                }
                cur1 = cur1.next;
            }
            return mm;
        }

        /// <summary>
        /// Complete linkage d(G,H)
        /// </summary>
        /// <param name="dmax"></param>
        /// <param name="n"></param>
        /// <param name="G"></param>
        /// <param name="iG"></param>
        /// <param name="H"></param>
        /// <param name="iH"></param>
        /// <returns></returns>
        private static double completelink(double[] dmax, int n, Cluster G, int iG, Cluster H, int iH)
        {
            double dmm = 0;
            Cluster cur = G;
            while (cur != null)
            {
                // dmax(g,H)
                if (dmax[n * cur.i + iH] > dmm) dmm = dmax[n * cur.i + iH];
                cur = cur.next;
            }
            cur = H;
            while (cur != null)
            {
                // dmax(h,G)
                if (dmax[n * cur.i + iG] > dmm) dmm = dmax[n * cur.i + iG];
                cur = cur.next;
            }
            return dmm;
        }

        #region R functions
        /// <summary>
        /// The lower triangle of the distance matrix stored by columns in a vector, say do. 
        /// If n is the number of observations, i.e., n &#60;- attr(do, "Size"), then for i &#60; j ≤ n, 
        /// the dissimilarity between (row) i and j is do[n*(i-1) - i*(i-1)/2 + j-i]. 
        /// The length of the vector is n*(n-1)/2, i.e., of order n^2. 
        /// </summary>
        /// <param name="distMatrix"></param>
        /// <returns></returns>
        public static double[] ToDist(double[][] distMatrix)
        {
            int n = distMatrix.Length;

            double[] dist = new double[n * (n - 1) / 2];

            for (int i = 1; i <= n; i++)
            {
                for (int j = i + 1; j <= n; j++)
                {
                    var index = (n * (i - 1) - i * (i - 1) / 2 + j - i) - 1;
                    dist[index] = distMatrix[i - 1][j - 1];
                }
            }
            return dist;
        }

        /// <summary>
        /// http://docs.rexamine.com/R-devel/sort_8c_source.html
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="nalast"></param>
        /// <returns></returns>
        private static int rcmp(double x, double y, bool nalast)
        {
            bool nax = double.IsNaN(x);
            bool nay = double.IsNaN(y);
            if (nax && nay) return 0;
            if (nax) return nalast ? 1 : -1;
            if (nay) return nalast ? -1 : 1;
            if (x < y) return -1;
            if (x > y) return 1;
            return 0;
        }

        /// <summary>
        /// http://docs.rexamine.com/R-devel/sort_8c_source.html
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="nalast"></param>
        /// <returns></returns>
        private static int icmp(int? x, int? y, bool nalast)
        {
            if (x == null && y == null) return 0;
            if (x == null) return nalast ? 1 : -1;
            if (y == null) return nalast ? -1 : 1;
            if (x < y) return -1;
            if (x > y) return 1;
            return 0;
        }

        /// <summary>
        /// http://docs.rexamine.com/R-devel/sort_8c_source.html#l00199
        /// </summary>
        /// <param name="x"></param>
        /// <param name="n"></param>
        private static void R_isort(int[] x, int n)
        {
            int v;
            bool nalast = true;
            int i, j, h;

            for (h = 1; h <= n / 9; h = 3 * h + 1) ;
            for (; h > 0; h /= 3)
                for (i = h; i < n; i++)
                {
                    v = x[i];
                    j = i;
                    while (j >= h && icmp(x[j - h], v, nalast) > 0)
                    { x[j] = x[j - h]; j -= h; }
                    x[j] = v;
                }
        }

        /// <summary>
        /// http://docs.rexamine.com/R-devel/sort_8c_source.html
        /// </summary>
        /// <param name="x"></param>
        /// <param name="indx"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        private static double[] rsort_with_index(double[] x, int[] indx, int n)
        {
            double v;
            int i, j, h, iv;

            for (h = 1; h <= n / 9; h = 3 * h + 1) ;
            for (; h > 0; h /= 3)
                for (i = h; i < n; i++)
                {
                    v = x[i]; iv = indx[i];
                    j = i;
                    while (j >= h && rcmp(x[j - h], v, true) > 0)
                    { x[j] = x[j - h]; indx[j] = indx[j - h]; j -= h; }
                    x[j] = v; indx[j] = iv;
                }
            return x;
        }
        #endregion

        private class Cluster
        {
            public int i;
            public Cluster next;
        }
    }
}
