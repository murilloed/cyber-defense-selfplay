import argparse,json
from pathlib import Path
import torch
from torch.distributions import Categorical
from .core import train,ActorCritic,reward
def main():
    p=argparse.ArgumentParser(); p.add_argument('--updates',type=int,default=30); p.add_argument('--seed',type=int,default=7)
    a=p.parse_args()
    if a.updates<1: p.error('updates must be positive')
    attacker,defender,checkpoints,history=train(a.updates,a.seed)
    torch.manual_seed(a.seed+100); states=torch.ones(2000,3); evaluations={}
    with torch.no_grad():
        dd=Categorical(logits=defender(states)[0])
        opponents=[('current',attacker),('uniform',None)]
        for i,weights in enumerate(checkpoints):
            m=ActorCritic(); m.load_state_dict(weights); opponents.append((f'checkpoint_{i}',m))
        for name,m in opponents:
            acts=torch.randint(0,3,(len(states),)) if m is None else Categorical(logits=m(states)[0]).sample()
            evaluations[name]={'mean_damage':float(reward(acts,dd.sample()).mean())}
        probs=torch.softmax(defender(torch.ones(1,3))[0],1)[0].tolist()
        best_response=max((1-probs[i])*(i+1) for i in range(3))
    report={'synthetic':True,'seed':a.seed,'updates':a.updates,'game':'one-step zero-sum MDP; abstract allocation',
            'defense_policy':probs,'evaluation':evaluations,'history':history,
            'worst_case_best_response_damage':best_response,
            'limitations':'One-state game; no claim of operational cybersecurity effectiveness.'}
    Path('artifacts').mkdir(exist_ok=True); Path('artifacts/report.json').write_text(json.dumps(report,indent=2))
    torch.save(defender.state_dict(),'artifacts/defender.pt'); print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2))
if __name__=='__main__': main()
