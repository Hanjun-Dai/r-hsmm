#ifndef KM_H
#define KM_H

#define KM_MAX_N 200
#define KM_INFTY 1073747823

class BipartiteGraph 
{
public:    
    BipartiteGraph(int num_nodes);
    int w[KM_MAX_N][KM_MAX_N], nlink[KM_MAX_N];

    void Init();

    int BestMatch();

private:
    bool dfs(int u);

    int ux[KM_MAX_N], uy[KM_MAX_N], slack[KM_MAX_N];
    int n;
    bool visx[KM_MAX_N], visy[KM_MAX_N];    
};

#endif