with open('GNU_RF_ENV/flowgraphs/step09_jittered_emitter.grc') as f1, open('GNU_RF_ENV/flowgraphs/step09b_jittered_emitter.grc') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    diff = [(i+1, l1.strip(), l2.strip()) for i, (l1, l2) in enumerate(zip(lines1, lines2)) if l1 != l2]
    print('Diff count:', len(diff))
    for d in diff[:20]:
        print(d)
