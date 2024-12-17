import stochastic_project.stochastic_parameter

prior = stochastic_project.stochastic_parameter.GaussianPrior(shape=(10,5))
prova1 = stochastic_project.stochastic_parameter.GaussianParticles(prior=prior)
prova1(n_particles=3)
prova2 = stochastic_project.stochastic_parameter.GaussianParameter(prior = prior)


for name, buff in prova2.named_buffers():
    print(name, buff)

for name, param in prova2.named_parameters():
    print(name, param)

for name, buff in prova1.named_buffers():
    print(name, buff)

for name, param in prova1.named_parameters():
    print(name, param)

print(prova1.flattened_particles)
print(prova1.flattened_particles.shape)