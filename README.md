# Slurm Tutorial

Welcome to the UKP Slurm Tutorial. This tutorial will give you a quick start in how to use the UKP/AIPHES/TK Slurm cluster and will cover the following steps:

- Connecting to the cluster
- Uploading data to the cluster
- Writing job execution scripts
- Requesting compute resources (CPU, RAM, GPUs)
- Submitting jobs
- Checking progress
- Receiving results

## What is SLURM?

Slurm is a job scheduling service. With Slurm, compute experiments are done by connecting to a headnode where one can submit a compute job into a queue of compute jobs. Slurm prioritizes compute jobs in a way that every user gets his/her share of the compute resources that he/she deserves. If a compute job has a high enough priority and a node (== compute server) with sufficient resources is free, Slurm will automatically start the compute experiment. Once a job finishes, an email notification is sent.

### Connecting to the cluster
This will be very short and fast tutorial of connecting SLURM and using LLM models. Please refer to [UKP Wiki Page of SLURM](https://wiki.ukp.informatik.tu-darmstadt.de/bin/view/Services/Campus/ComputeCluster) and [FAQ](https://wiki.ukp.informatik.tu-darmstadt.de/bin/view/Services/Campus/ComputeClusterFAQ).

While connecting TU VPN, you can connect to SLURM from the terminal via:

`ssh slurm.ukp.informatik.tu-darmstadt.de`

When you are in SLURM you should work only in your workspace which you can reach via:

`cd /ukp-storage-1/your_name`

You need to work only in this workspace.


## Creating your python environment

Then, you need to create a virtual environment or conda. What I prefer is miniconda. You can install it via [here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

Make all installations after "srun --pty bash -i"

After the installation you can create your project environment by:

`conda create -n your_environment_name`

Then, you need to install some packages (ON A COMPUTE NODE WITHOUT GPU, NOT ON THE LOGIN NODE):

to connect to a login node run:

`srun --pty bas

Then, activate your environment:

`conda activate your_environment_name`

Now you can install the packages:

For pytorch:

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

For transformers:

`pip install transformers`

For tokenizers:

``

You can install other packages according to your needs as well.

## Llama-2 Chat Simple Usage Snipped

* `simple_llama_2.py` is a small code snippet as an example of how to run a pretrained model for inference purposes in our server.

To be able to run our llama 2 tutorial you need `torch`, `transformers` libraries. If you want to load model in 8 bits, you will need `accelerate` and `bitsandbytes` libraries as well. You can do this by activating your environment on a compute node without GPU as described above.

### Uploading data (you can skip this part for now)

The headnode and all compute nodes share one network file system. It is mounted under `/ukp-storage-1` . This means that you just need to copy the data once to the network share, not to every single node. Jobs can then see the data no matter on which compute server the job runs. For more information about the cluster storage, please refer to the [Compute Cluster Storage documentation](https://wiki.ukp.informatik.tu-darmstadt.de/Services/Campus/ComputeClusterStorage) .

### Copying data to the cluster

Every user of the cluster has their own personal folder on the cluster storage at `/ukp-storage-1/<username>` where data and code should be stored. This folder should already exist when you log in for the first time. Please **do not** place code or data in `/tmp` or `/home` on the headnode!
In order to use your existing code and data, it has to be copied to the cluster storage first. For that, `scp` and `rsync` can be used from the command line. We will use `scp` as an example for this tutorial. 

For that, copy the file `tensorflow_mnist.py` to your local computer. Then, transfer it to the network storage by executing (replace `user` again with your last name)

    $ scp simple_llama_2.py user@slurm.ukp.informatik.tu-darmstadt.de:/ukp-storage-1/user

Using `scp` is only one way of transferring data to the cluster storage. You can also e.g. clone git repositories directly into your user folder on the storage while logged in to the headnode.


* There are several models, you can look at them in this directory `/storage/ukp/shared/shared_model_weights/`. For Llama 2 models, you can reach them via `/storage/ukp/shared/shared_model_weights/models--llama-2-hf/`. For initial start, you can start with 70B-Chat model. Its directory is provided in the code. Please do not re-download those models. 


* Here is our shared model [page](https://wiki.ukp.informatik.tu-darmstadt.de/bin/view/UKP/SharedModels) in wiki where you can find beneficial information about how to find and use models in our server. 


* Changing tokenizer settings may result in some inconsistencies of the results, especially for batch tokenization. For reproducibility, this is an undesirable situation.


* For generation parameters, you can utilize this [page](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#generation) of Huggingface.


* Llama models output all previous prompts along with its answer. It's better to remove given prompt while inspecting the output.


* We have also small LLM [github repository](https://git.ukp.informatik.tu-darmstadt.de/kuznetsov/llm_snippets/-/tree/main/) for different information, you can give a look.


* You need to adjust the code such that sampled questions provided to model get answers. 

## Submit Your Job

* For computing in GPUs, you need to submit your jobs to computing nodes. Please do not try to run the code directly. 


* You can use run.sh file where you can adjust several settings for this purpose. Please change information like project path, your environment name, mail address etc.
  * Output path is not your output path for your code. It will output some information about your code's running status.
  * You don't have to change other parameters I adjusted them. If you are curious about them please check the corresponding wiki pages that I provided above.


* You can submit your job by:

`sbatch run.sh`

Then, you will be given a job number. To monitor your code's output you can run: 

`sattach your_job_id.0` (in general it does not end with 0 (zero) but you are just running single code. So don't change it.)

Depending on the job traffic your job may not be implemented immediately. You can monitor status of your jobs as running:

`squeue --me` If your job is running, it is labeled as (R). If it's pending, the label is (PD) along with the pending reason.

All sources of the lab are shared, so please use them responsibly. Down below you can find some more information about the potential parameters you can provide when submitting a job.

You can find all of these information and more in the corresponding wiki pages. I just provided the most essential ones.

## Specifying sbatch parameters

We first start with the job parameters which, among others, specify which resources the job needs or where the output should be written. Add the following template to the top of `slurm_script.sh`. We will fill the placeholders in in a moment:

    #!/bin/bash
    #
    #SBATCH --job-name=tensorflow_mnist
    #SBATCH --output=/ukp-storage-1/USERNAME/res.txt
    #SBATCH --mail-user=YOUR-EMAIL-ADDRESS
    #SBATCH --mail-type=ALL
    #SBATCH --account=ACCOUNT-NAME
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=2
    #SBATCH --mem=8GB
    #SBATCH --gres=gpu:1

The `job-name` parameter sets the job name. This name will show up when you run `squeue` on the headnode to check the status of your jobs.

`output` specifies where to save `stdout` and `stderr` to. Please fill in your username for `USERNAME`. These can be also split into two files, default is that all goes into `output`. It is recommended to always use absolute paths. You can also use placeholders in these file names to create a different file for different runs. As an example, this could look like `#SBATCH --output=/ukp-storage-1/user/myjob.%j.%N.out`. A list of all available placeholders can be found in the [sbatch documentation](https://slurm.schedmd.com/sbatch.html#SECTION_<B>filename-pattern</B>).

`mail-user` specifies where to send mails related to your job to. The `mail-type` describes which mails you want to get, e.g. when your job actually starts, crashes or finishes. Please enter an email address of yours to receive these kinds of notifications.

For `account`, you need to specify one of the accounts of your Slurm user. Slurm distinguishes between "users" (human beings) and "accounts" (like bank accounts) which record resource usage. One user can have multiple accounts. Find out which accounts are available to you by running `sshare -U` on the headnode. Enter the name of one of your accounts in the template.

`partition` determines into which job queue your job will be submitted. For this example, we will use the default partition. For larger/longer jobs, you will want to read the [cluster documentation](https://wiki.ukp.informatik.tu-darmstadt.de/Services/Campus/ComputeCluster) to learn about other partitions.

There is one python command in our script we want to run (`ntasks=1`) and it should use two CPUs (`cpus-per-task=2`). Also, we only need 8GB of RAM `mem=8GB`. If you do not specify these, then defaults are used. These are described in the [cluster documentation](https://wiki.ukp.informatik.tu-darmstadt.de/Services/Campus/ComputeCluster).

`gres` desribe additional resources requirements for your job. We currently only use this for GPUs. You can either request any available GPU model via `#SBATCH --gres=gpu:<number>` or a specific model `#SBATCH --gres=gpu:<gpu_name>:<number>`. Which models are available can be seen via `sinfo --Format=gres`. 
When requesting resources, keep in mind that they are billed to your account. Faster CPUs and GPUs are more "expensive" than slower ones and will impact the queueing priority of your future jobs.

The official [sbatch](https://slurm.schedmd.com/sbatch.html) documentation describes all parameters that can be used. More examples can be found on the websites of the [TU Darmstadt Lichtenberg](https://www.hhlr.tu-darmstadt.de/hhlr/arbeit_auf_dem_cluster/arbeit_mit_lsf_1/index~1.de.jsp) or the [Leipzig Rechenzentrum](https://doku.lrz.de/display/PUBLIC/Example+parallel+job+scripts+on+the+Linux-Cluster), or at [Univ Cambridge](https://www.ch.cam.ac.uk/computing/slurm-usage).

### Specifying your computation

Add the following to the end of `slurm_script.sh` and adjust `USERNAME`:

    source /ukp-storage-1/USERNAME/my_venv/bin/activate
    module purge
    module load cuda/10.0
    python /ukp-storage-1/USERNAME/tensorflow_mnist.py

This will activate the virtual environment we specified before, load a specific CUDA version which is required for tensorflow and then run the script. To learn more about modules, please refer to the [compute cluster FAQ](https://wiki.ukp.informatik.tu-darmstadt.de/Services/Campus/ComputeClusterFAQ).


