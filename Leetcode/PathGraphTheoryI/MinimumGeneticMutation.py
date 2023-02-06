class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: list[str]) -> int:
        if endGene not in bank:
            return -1
        bank = set(bank)
        myque = collections.deque([(startGene, 0)])
        while(myque):
            gene, hops = myque.popleft()
            if gene == endGene:
                return hops
            for i in range(len(gene)):
                for j in ['A', 'C', 'G', 'T']:
                    if j == gene[i]:
                        continue
                    new_gene = gene[:i] + j + gene[i + 1:]
                    if new_gene in bank:
                        bank.remove(new_gene)
                        myque.append((new_gene, hops + 1))
        return -1