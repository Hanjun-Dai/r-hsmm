#include "km.h"
#include <cstring>

BipartiteGraph::BipartiteGraph(int num_nodes)
{
		n = num_nodes;
}

void BipartiteGraph::Init()
{       
        memset(ux, 0, sizeof(ux));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (w[i][j] > ux[i])
                    ux[i] = w[i][j];
        memset(uy, 0, sizeof(uy));
        for (int i = 0; i < n; ++i)
        	nlink[i] = -1;
}

bool BipartiteGraph::dfs(int u) {
        int t;
        visx[u] = true;
        for (int i = 0; i < n; i++)
            if (! visy[i]) {
                t = ux[u] + uy[i] - w[u][i];
                if (t == 0) {
                    visy[i] = true;
                    if ((nlink[i] == -1) || (dfs(nlink[i]))) {
                        nlink[i] = u;
                        return(true);
                    }
                }
                else
                    if (t < slack[i])
                        slack[i] = t;
            }
        return(false);
}

int BipartiteGraph::BestMatch() {
        int i, j;
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                slack[j] = KM_INFTY;
            while (1) {
                memset(visx, false, sizeof(visx));
                memset(visy, false, sizeof(visy));
                if (dfs(i))
                    break;
                int d = KM_INFTY;
                for (j = 0; j < n; j++)
                    if ((! visy[j]) && (slack[j] < d))
                        d = slack[j];
                for (j = 0; j < n; j++) {
                    if (visx[j])
                        ux[j] -= d;
                    if (visy[j])
                        uy[j] += d;
                    else
                        slack[j] -= d;
                }
            }
        }

        int ans = 0;
        for (int i = 0; i < n; i++)
            ans += w[nlink[i]][i];
        return(ans);
}