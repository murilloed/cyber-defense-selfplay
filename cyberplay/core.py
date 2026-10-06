import torch
from torch import nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__(); self.body=nn.Sequential(nn.Linear(3,24),nn.Tanh())
        self.policy=nn.Linear(24,3); self.value=nn.Linear(24,1)
    def forward(self,x):
        h=self.body(x); return self.policy(h),self.value(h).squeeze(-1)

def reward(attack,defend):
    # Matching defence blocks the attack; otherwise asset weight is lost.
    weights=torch.tensor([1.,2.,3.],device=attack.device)
    return torch.where(attack==defend,torch.zeros_like(attack,dtype=torch.float),weights[attack])

def ppo_step(model,opt,states,actions,old_log,returns,epochs=3):
    with torch.no_grad(): _,old_v=model(states)
    advantages=returns-old_v
    advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-6)
    for _ in range(epochs):
        logits,values=model(states); dist=Categorical(logits=logits)
        ratio=torch.exp(dist.log_prob(actions)-old_log)
        unclipped=ratio*advantages; clipped=ratio.clamp(.8,1.2)*advantages
        loss=-torch.minimum(unclipped,clipped).mean()+.5*(values-returns).square().mean()-.01*dist.entropy().mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),.5); opt.step()
    return float(loss.detach())

def train(updates=30,seed=7):
    torch.manual_seed(seed); torch.set_num_threads(1)
    attacker,defender=ActorCritic(),ActorCritic()
    ao=torch.optim.Adam(attacker.parameters(),lr=.003); do=torch.optim.Adam(defender.parameters(),lr=.003)
    states=torch.ones(128,3); history=[]; checkpoints=[]
    for t in range(updates):
        with torch.no_grad():
            ad=Categorical(logits=attacker(states)[0]); dd=Categorical(logits=defender(states)[0])
            a,d=ad.sample(),dd.sample(); al,dl=ad.log_prob(a),dd.log_prob(d)
            r=reward(a,d)
        la=ppo_step(attacker,ao,states,a,al,r)
        ld=ppo_step(defender,do,states,d,dl,-r)
        history.append({'update':t,'mean_attack_return':float(r.mean()),'attack_loss':la,'defense_loss':ld})
        if t in (0,updates//2): checkpoints.append({k:v.clone() for k,v in attacker.state_dict().items()})
    return attacker,defender,checkpoints,history
