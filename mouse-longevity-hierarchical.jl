# -*- coding: utf-8 -*-
# ---
# title: Bayesian hierarchical modeling of mouse longevity data
# author: Śaunak Sen
# date: 2023-05-02
# weave_options:
#   line_width : 400
# jupyter:
#   jupytext:
#     formats: ipynb,jl
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Julia-16-threads 1.8.5
#     language: julia
#     name: julia-16-threads-1.8
# ---

# In this note we perform a Bayesian hiearchical modeling of mouse longevity data. We show how
# such modeling can be performed in Julia using the Turing.jl and associated packages.  We also
# discuss data analysis choices and compare the output from different models. The main analysis
# is a Bayesian two-level model assuming normally distributed residuals. We tweak this model
# by allowing flat-tailed residuals, and by allowing the residual variance to differ by cluster.
#
# The computations here were performed using Julia with 16 threads used for computation.
# Choosing the tuning parameters and starting values for the computations was chellenging.

# # BXD longevity data 
#
# This data comes from a mouse longevity study conducted by Rob Williams and colleagues. About 1700 mice from about 110 mouse inbred strains were followed until their death. About half the mice were fed a high-fat diet (HF) and the rest were fed the control or "chow" diet (CD). The mouse strains are derived from a cross between the B6 and DBA mouse strains.  In that sense, the strains are a sample from the population of all possible strains that can be made from crossing those two strains. Mice from the same strain are expected to have more similar longevity.  In this study mice (first level units) are clustered within strains (second level units).
#
# We want to know the impact of strain genetic background and diet on longevity, and if the effect of diet depends on strain.

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using MixedModels, GLM, CSV, DataFrames, Weave
using FreqTables, CategoricalArrays, StatsPlots, StatsModels, Statistics, StatsBase
using LinearAlgebra, Turing, Distributions, Random, Optim

gr()
# to operate in headless mode      
# https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
ENV["GKSwstype"] = "100"
ENV["COLUMNS"] = "200"
set_chunk_defaults!(:line_width => 400)
set_chunk_defaults!(:term => true)

# We read in the data and look at the column names.

lifespan = CSV.read("/home/sen/uthsc/data/bxd/longevity/AgingBXD_LongevityData_3Feb2023.csv",
    DataFrame,header=4);
names(lifespan)

first(lifespan) |> (x->show(x,allcols=true))

# We rename a few columns for convenience and summarize the data by diet and strain.

rename!( lifespan, 7=>:lifetime, 4=>:strain, 6=>:diet );

strainN = combine(groupby(lifespan,
        r"strain|diet"), :lifetime => (x-> length(collect(skipmissing(x)))) => :n);
strainN4 = unstack(strainN,:strain,:diet,:n) |>
       (df->subset(df,:HF => (h-> h .> 3),skipmissing=true)) |> 
       (df->subset(df,:CD => (c-> c .> 3),skipmissing=true)) |>
       (df->subset(df,:strain => (x->startswith.(x,"BXD"))));

lifespanN4 = semijoin(lifespan,strainN4,on=:strain);

strainDietTableN4 = combine(groupby(lifespanN4,r"^diet|^strain"),  
    :lifetime => (x->length(collect(skipmissing(x)))) => :n, 
    :lifetime => (x->mean(skipmissing(x))) => :mean, 
    :lifetime => (x->std(skipmissing(x))) => :sd,
    :lifetime => (x->median(skipmissing(x))) => :median);

strainDietTableN4 |> (df->first(df,8))

strainDietTableN4CD = subset(strainDietTableN4,:diet => (x->x.=="CD")) |> 
     (df->sort(df,:median))
strainDietTableN4HF = subset(strainDietTableN4,:diet => (x->x.=="HF")) |>
      (df->sort(df,:median));
strainDietTableN4Wide = leftjoin(strainDietTableN4HF,strainDietTableN4CD,on=:strain,
    renamecols="_HF"=>"_CD",order=:left);

first(strainDietTableN4Wide,4)

# ## Dotplot of longevity by strain and diet

strainDietTableN4Wide.medianOrder = 1:nrow(strainDietTableN4Wide);
lifespanN4 = leftjoin(lifespanN4,select(strainDietTableN4Wide,
        :strain,:medianOrder),on=:strain);

using Plots.PlotMeasures
@df subset(lifespanN4,:diet=>(x->x.=="HF")) dotplot(:medianOrder.+0.1,:lifetime,size=(1000,500),side=:right,mode=:none,
    markersize=4,markershape=:rtriangle,palette=:Dark2_4,markeralpha=0.3,bottom_margin=10mm,left_margin=5mm,label="HF",ylabel="Longevity (days)")
@df subset(lifespanN4,:diet=>(x->x.=="CD")) dotplot!(:medianOrder.-0.1,:lifetime,side=:left,mode=:none,
    markersize=4,markershape=:ltriangle,markeralpha=0.3,markercolor=2,label="CD")
@df strainDietTableN4Wide plot!(:medianOrder,:median_HF,
    markersize=3,markershape=:circle,markeralpha=0.7,linecolor=1,markercolor=1,label="median HF")
@df strainDietTableN4Wide plot!(:medianOrder,:median_CD,
    markersize=3,markershape=:circle,markeralpha=0.7,linecolor=2,markercolor=2,label="median CD")
@df strainDietTableN4Wide xticks!(:medianOrder,:strain,xrotation=90)

# ## Histogram of lifetimes

histogram(lifespanN4.lifetime,label="",palette=:tab10)

# ### Violin plot of lifetimes

@df subset(lifespanN4,:diet=>(x->x.=="HF")) violin(ones(length(:lifetime)),:lifetime,label="HF",side=:left,xtick=:none,palette=:Dark2_4,alpha=0.4)
@df subset(lifespanN4,:diet=>(x->x.=="CD")) violin!(ones(length(:lifetime)),:lifetime,label="CD",side=:right,alpha=0.4,size=(400,500))

# ## Scatterplot of strain means by diet

scatter(strainDietTableN4.mean[strainDietTableN4.diet.=="HF"],strainDietTableN4.mean[strainDietTableN4.diet.=="CD"],
    xlim=(300,1000),ylim=(300,1000),label="",xlab="HF diet mean",ylab="CD diet mean",size=(500,400))
plot!(x->x,label="")

@df strainDietTableN4 histogram(log.(:sd.^2),label="")

# # Linear mixed model
#
# We now fit a linear mixed model with a random effect of strain, and a fixed effect of diet.
#
# If $y_{ij}$ is the $j$-th observation in the $i$-th
# strain, then we can write the linear model
#
# $$y_{ij} = \mu + \delta t_{ij} + \beta_i + \epsilon_{ij},$$
#
# where $\mu$ is a fixed effect of the overall mean and $\beta$'s are random effects with mean
# zero and variance $\tau^2$. The idea now is to assume that the mean for each strain is now
# random with mean $\mu$ and variance $\tau^2$. The residual error denoted by the $\epsilon$'s 
# has variance $\sigma^2$. The variable $t_{ij}$ indicates whether the individual was given chow
# or high-fat diet.

outLMM = fit(MixedModel,@formula(lifetime~1+diet+(1|strain)),
    lifespanN4,contrasts=Dict(:strain => StatsModels.FullDummyCoding(),
        :diet => EffectsCoding()));

outLM = fit(LinearModel,@formula(lifetime~1+diet+strain),
    lifespanN4,contrasts=Dict(:strain => EffectsCoding(),:diet => EffectsCoding()));

show(outLMM)

# # Bayesian version of LMM
#
# We now fit the Bayesian version of the LMM above.  It has the same form, with the
# addition of priors for the unknown parameters.
#
# We chose the priors as follows. We follow the current recommendation of using
# _weakly informative_ priors instead of "non-informative" priors. These have the
# advantage of guaranteeing that the posterior distributions are proper. They
# can also be used to guide the posterior to explore relevant portions of the
# parameter space.
#
# - $\mu$ (overall mean): Normal with mean equal to the overall empirical mean, and twice the empirical sd.
# - $\delta$ (diet effect): Normal with zero mean and save sd as overall mean.
# - $\sigma$ (residual sd): Half Cauchy with scale parameter equal to empirical sd.
# - $\tau$ (strain effect sd): Same as residual sd.
# - $\beta$'s (strain effects): Normal with zero mean, and sd $\tau$.

# +
@model function linearMixedModel(y,strain,diet)
    stdy = std(y)
    nstrains = length(unique(strain))
    strainNames = sort(unique(strain))
    
    ndiets = length(unique(diet))
    dietNames = sort(unique(diet))
    
    μ ~ Normal(mean(y),stdy*2)
    δ ~ Normal(0.0,stdy*2)
    
    σ ~ truncated(Cauchy(0,stdy);lower=0)
    τ ~ truncated(Cauchy(0,stdy);lower=0)

    β ~ MvNormal( zeros(nstrains), I*τ^2 )
        
    strainEffectDict = Dict(zip(strainNames,β))
    strainMeans = (x->strainEffectDict[x]).(strain)
    dietEffectDict = Dict(zip(dietNames,[-δ/2,δ/2]))
    dietEffects = (x->dietEffectDict[x]).(diet)        
    means = μ .+ strainMeans .+ dietEffects
    
    y ~ MvNormal(means,I*σ^2)
    
end
# -

nstrains = length(unique(lifespanN4.strain))
meany = mean(lifespanN4.lifetime)
bayesLMM = @df lifespanN4 linearMixedModel(:lifetime,Vector(:strain),Vector(:diet));
bayesLMMStart = [ meany; -80.0; 130.0; 100.0; rand(Normal(0,50),nstrains) ];

Random.seed!(123);
@time bayesLMMPostSample = sample(bayesLMM, 
    Gibbs(NUTS(200,0.65,init_ϵ=0.02,:μ,:δ,:σ,:τ),NUTS(200,0.65,init_ϵ=25.0,:β)),
    MCMCThreads(),1000,16,
    init_params = fill(bayesLMMStart,16));

summarystats(bayesLMMPostSample) |> DataFrame

plot(bayesLMMPostSample[:,1:5,:])

# ### Comparing LM estimates to LMM estimates
#
# We can see that the LMM estimates are similar but shrunk towards zero.

# +
# ### Comparing Bayes, LMM, and LM methods

bayesRE = DataFrame(summarystats(bayesLMMPostSample))[5:end,1:3];
bayesRE.parameters = (sort(unique(lifespanN4.strain)));

lmmRE = raneftables(outLMM)[1] |> collect |> (x->DataFrame(x,:auto))
lmmRE.se = sqrt.(condVar(outLMM)[1][1,1,:]);
rename!(lmmRE,2=>"blup",1=>"strain");

effects = leftjoin(bayesRE,lmmRE,on=(:parameters=>:strain));

outLM = fit(LinearModel,@formula(lifetime~1+diet+strain),
    lifespanN4,contrasts=Dict(:strain => EffectsCoding(),:diet => EffectsCoding()));
lmEst = DataFrame(coeftable(outLM))[3:end,1:3];

transform!(lmEst,:Name => (s->replace.(s,r"^strain: " => s"")) => :Name)
rename!(lmEst,1=>:strain);

effects = leftjoin(effects,lmEst,on=(:parameters=>:strain));
sort!(effects,:parameters);
rename!(effects,1=>:strain,2=>:bayesEst,
    3=>:bayesSE,4=>:lmmEst,5=>:lmmSE,
    6=>:lmEst,7=>:lmSE);
effects.lmEst[1] = -sum(effects.lmEst[2:end]);
# -

@df effects scatter(:lmEst,:lmmEst,size=(500,400),label="",
    xlabel="LM estimate",ylabel="LMM estimate",framestyle=:box)
plot!(x->x,label="")

# ### Comparing LMM and Bayes estimates
#
# We can see that they are virtually identical.

@df effects scatter(:lmmEst,:bayesEst,size=(500,400),label="",
    xlabel="LMM estimate", ylabel="Bayes estimate", framestyle=:box)
plot!(x->x,label="")

# # Relaxing normality assumption
#
# If we are concerned about outliers, we may want to relax the normality 
# assumption for the lifetimes.  This would be difficult with the LMM in
# that no ready implementations are available.  We can do that in the
# Bayesian version, by assuming that the lifetimes have a t-distribution
# with 4 degrees of freedom.

# ## Density function comparison with same scale parameter
#
# We can see that the t-distribution with 4 df has flatter tails compared to a Normal distribution
# with the same (unit) scale parameter. This is the smallest df for which the t-distribution has
# the first 3 moments.

plot(TDist(4),label="t, df=4")
plot!(Normal(),label="Normal")
plot!(x->0,color=:black,label="")

# The function to construct the joint distribution is virtually identical to the
# previous one exceot for the very last line, where we make the outcome (lifetime)
# conditional on the random effects to have a t-distribution with 4df.

# +
@model function linearMixedModelTDist(y,strain,diet)
    stdy = std(y)
    nstrains = length(unique(strain))
    strainNames = sort(unique(strain))
    
    ndiets = length(unique(diet))
    dietNames = sort(unique(diet))

    μ ~ Normal(mean(y),stdy*2)
    δ ~ Normal(0,stdy*2)

    σ ~ truncated(Cauchy(0,stdy);lower=0)
    τ ~ truncated(Cauchy(0,stdy);lower=0)

    β ~ MvNormal( zeros(nstrains), I*τ^2 )
        
    strainEffectDict = Dict(zip(strainNames,β))
    strainMeans = (x->strainEffectDict[x]).(strain)
    dietEffectDict = Dict(zip(dietNames,[-δ/2,δ/2]))
    dietEffects = (x->dietEffectDict[x]).(diet)        
    means = μ .+ strainMeans .+ dietEffects
    
    # y ~ arraydist( Normal()*σ .+ means )
    # Turing.@addlogprob! -nstrains*log(σ) + sum( (x->logpdf( TDist(12), x )).((y.-means)./σ) )
    y ~ arraydist( TDist(4)*σ .+ means )
    
end
# -

# To help the sampling, we initialize with starting values similar to that from the normal
# residual LMM.

bayesLMMTDist = @df lifespanN4 linearMixedModelTDist(:lifetime,:strain,:diet);
bayesLMMTDistStart = [ meany; -80.0; 130.0; 100.0; rand(Normal(0,50),nstrains) ];

# We run NUTS with the stepsize parameter close to that for the normal model.  The idea
# is that we expect the overall shape to be similar.

Random.seed!(123);
@time bayesLMMTDistPostSample = sample(bayesLMMTDist, 
    Gibbs(NUTS(1000,0.65,init_ϵ=0.02,:μ,:δ,:σ,:τ),NUTS(1000,0.65,init_ϵ=25.0,:β)),
    MCMCThreads(),1000,16,
    init_params = fill(bayesLMMTDistStart,16));

summarystats(bayesLMMTDistPostSample[:,:,:]) |> DataFrame

# The chains seem to have converged, and different threads give about the same distribution.

plot(bayesLMMTDistPostSample[:,1:7,:])

# +
bayesTDistRE = DataFrame(summarystats(bayesLMMTDistPostSample[:,:,:]))[5:end,1:3];
bayesTDistRE.parameters = (sort(unique(lifespanN4.strain)));

effects = leftjoin(bayesTDistRE,effects,on=(:parameters=>:strain))
rename!(effects,:parameters=>:strain,:mean=>:bayesTDistEst,:std=>:bayesTDistSE)
# -

# ## Comparing LMM to Bayes estimates with t-distributed residuals
#
# We see that there are some differences from the LMM (unlike the normal
# residual Bayes model).  Overall, the estimates are still quite similar.

@df effects scatter(:lmmEst,:bayesTDistEst,size=(500,400),label="",
    xlabel="LMM estimate", ylabel="Bayes t-dist estimate", framestyle=:box)
plot!(x->x,label="")

# # Allowing strain variances to vary
#
# Next, we make a second change in the model.  We allow the residuals to be normally distributed,
# but we allow the variances to depend on the strain.  We use a hiearchical model for the variances
# as follows.
#
# - The strain variance for the $i$-th strain given the random effect is lognormal with 
#   location parameter $\theta + \phi_i$ and scale parameter $\kappa$.
# - The scale parameter $\kappa$ has a unit half Cauchy distribution.

# +
@model function linearMixedModelVar(y,strain,diet)
    stdy = std(y)
    nstrains = length(unique(strain))
    strainNames = sort(unique(strain))
    
    ndiets = length(unique(diet))
    dietNames = sort(unique(diet))
    
    # std of log within strain stds
    κ ~ truncated(Cauchy(0,1);lower=0)     
    # overall mean
    μ ~ Normal(mean(y),stdy*4)
    # diet effect
    δ ~ Normal(0,stdy*4)
    # std of strain random effects
    τ ~ truncated(Cauchy(0,stdy);lower=0)
    # within strain stds
    θ ~ Normal(log(stdy),4)
    ϕ ~ MvNormal( zeros(nstrains), I*κ^2 )
    # within strain means
    β ~ MvNormal( zeros(nstrains), I*τ^2 )
       
    strainEffectDict = Dict(zip(strainNames,β))
    strainVarDict = Dict(zip(strainNames,exp.(2.0.*(θ.+ϕ))))
    
    strainMeans = (x->strainEffectDict[x]).(strain)
    strainVars = (x->strainVarDict[x]).(strain)

    dietEffectDict = Dict(zip(dietNames,[-δ/2,δ/2]))
    dietEffects = (x->dietEffectDict[x]).(diet)        
    means = μ .+ strainMeans .+ dietEffects
    
    # y ~ arraydist( (x->Normal(x,σ)).(means) )
    y ~ MvNormal(means,Diagonal(strainVars))
    
end
# -

bayesLMMVar = @df lifespanN4 linearMixedModelVar(:lifetime,Vector(:strain),Vector(:diet));

# We use starting values from the LMM, and for the scale parameter we use the two-step modeling of
# the variances (see sections below).

# +
# @time lmmVarPostSample = sample(lmmVar,
#      Gibbs(NUTS(100,0.65,:β),NUTS(100,0.65,:θ),NUTS(100,0.65,:τ0,:τ2,:δ,:μ)),
#      MCMCThreads(),1000,16,init_params = fill(lmmVarMAP.values.array,16));
stdy = std(lifespanN4.lifetime)
meany = mean(lifespanN4.lifetime)
nstrains = length(unique(lifespanN4.strain))

bayesLMMVarStart = [ 0.4; meany; -80.0; 100.0; log(130.0); rand(Normal(0,0.1),nstrains); rand(Normal(0,50),nstrains) ];
# -

Random.seed!(123);
@time bayesLMMVarPostSample = sample(bayesLMMVar,
   Gibbs(NUTS(200,0.65,init_ϵ=0.02,:μ,:θ,:δ,:τ,:κ),NUTS(200,0.65,init_ϵ=25.0,:β),NUTS(200,0.65,init_ϵ=0.02,:ϕ)),
   MCMCThreads(),1000,16,init_params = fill(bayesLMMVarStart,16));

summarystats(bayesLMMVarPostSample) |> DataFrame

# The chain takes a few iterations to converge, but overall is mixing well; distributions across parallel chains
# are similar.

plot(bayesLMMVarPostSample[:,[1:6;70:72;151:153],:])

# +
bayesVarRE = DataFrame(summarystats(bayesLMMVarPostSample[101:1000,:,:]))[(74+6):end,1:3];
bayesVarRE.parameters = (sort(unique(lifespanN4.strain)));

effects = leftjoin(bayesVarRE,effects,on=(:parameters=>:strain))
rename!(effects,:parameters=>:strain,:mean=>:bayesVarEst,:std=>:bayesVarSE);
# -

# ## Comparing LMM to Bayes estimates with heteroscedastic residuals
#
# Heteroscedasticity appears to have little effect on the strain mean estimates
# from the Bayes model.

@df effects scatter(:lmmEst,:bayesVarEst,size=(500,400),label="",
    xlabel="LMM estimate", ylabel="Bayes heteroscedastic estimate", framestyle=:box)
plot!(x->x,label="")

# # Two-step models for the variances
#
# Before I conducted the heteroscedastic analysis, I did a two-step analysis
# for the variance heterogeneity to get a rough idea of the variability in the
# within-strain variances.
#
# My starting point was to look at the log of the variance using a LMM, but
# quickly realized when I started writing the function, that the standard
# machinery would not work (see below for function that I did not complete
# writing).  The reason is that the log transformation is a variance stabilizing
# transform for a variance, and in that case, the variance (of the sample variance)
# is _known._

# ### Modeling the within-strain variances

# ```julia
# function shrinkVar(v::Vector{Float64},n::Vector{Int64})
#    
#     logv = log.(v)
#     logvmean = mean(logv)
#     ϕ = 2.0./(n.-2)
#     
#     out = fit(MixedModel,@formula(logv~1+(1|strain)))
#     λ = ϕ./(ϕ.+τ)
#     logvShrink = (1.0.-λ) .* logv + λ.*logvmean
# end
# ```

# ## Histogram of log strain variances

histogram(2*log.(strainDietTableN4.sd),bins=20,label="")

# ## Approximate distribution of log variances
#
# It can be shown that the variance of the log sample variances is approximately 
# $2/(n-2)$. We plot here the variance of the log of the variance from a normally
# distributed outcome with sample size. We can see this from the figure below, that
# the approximation is very good.

# ### Plot of variance of log of strain variance

n = (3:20)
logVarN = n .|> (n->rand(Chisq(n-1),1_000_000)) .|> 
       (x->log.(x)) .|> 
       var
scatter(n,logVarN,ylabel="variance of log strain variance",
    xlabel="sample size(n)",label="Actual variance")
plot!(n,2 ./ (n.-2),label="2/(n-2)")

# We can use this to model the log variances in a two-step Bayesian model.
# Note that this has the same form as the earlier Bayesian LMM, just that it is on
# the log strain variances, and we know the variance.

@model function logvar(logv::Vector{Float64},n::Vector{Int64})
    k = length(logv)
    τ ~ truncated(Cauchy(0,1);lower=0)
    μ ~ Normal(mean(logv),10)
    β ~ MvNormal(zeros(k),I*τ^2)
    logv ~ MvNormal(μ.+β,Diagonal(2.0./(n.-2)))
end

logVar = @df strainDietTableN4 logvar(log.(:sd.^2),:n);

@time logVarPostSample = sample(logVar,NUTS(200,0.65,init_ϵ=0.02),MCMCThreads(),1000,16);

summarystats(logVarPostSample) |> DataFrame

plot(logVarPostSample[:,1:4,:])

# These results suggest that although there is heterogeneity among the strains (in their
# variance) the magnitude is small.  However, this modeling is based on an approximate
# result, so we use a more direct Bayesian model that assumes that the strain variances
# have a $\chi^2$ distribution.  This may be more palatable than assuming that the log
# strain variances have a normal distribution.

# ## Modeling the variances directly
#
# Instead of relying on the variance stabilizing transform, we can
# directly model the variation in the variances. We assume that the
# strain variances have a $\chi^2$ distribution with $n-1$ degrees of 
# freedom.

@model function shrinkvar(v::Vector{Float64},n::Vector{Int64})
    k = length(v)
    τ ~ truncated(Cauchy(0,1);lower=0)
    μ ~ Normal(mean(log.(v)),10)
    β ~ MvNormal(zeros(k),I*τ^2)
    v ~ arraydist( exp.(μ .+ β) .* (df->Chisq(df-1)).(n) )
end

sv = @df strainDietTableN4 shrinkvar((:sd.^2),:n);

@time svPostSample = sample(sv,NUTS(200,0.65,init_ϵ=0.002),MCMCThreads(),1000,16);

summarystats(svPostSample) |> DataFrame

plot(svPostSample[:,1:5,:])

# The overall conclusion is the same as in the approximate analysis above.  We used
# the value of $\tau$ here to suggest the starting value for $\kappa$ in the
# heterscedastic analysis for the Bayesian LMM above.
