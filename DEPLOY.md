# NeuroScan NG: Hosting Guide

Two paths. Path A is free and takes about 20 minutes, good for your defense. Path B is a real server that keeps records, free for a year with the GitHub Student Pack.

Both start the same way: put the code on GitHub.

## Step 0: Push to GitHub (both paths)

On your machine, inside the unzipped `neuroscan-ng` folder:

```bash
git init
git add .
git commit -m "NeuroScan NG: Alzheimer's MRI detection system"
```

Create an empty repo on github.com (New repository, name it `neuroscan-ng`, no README), then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/neuroscan-ng.git
git branch -M main
git push -u origin main
```

If you have trained model files from the Colab notebook, copy them into `model/` before committing. The `.keras` file is large, so install git-lfs first:

```bash
git lfs install
git lfs track "model/*.keras"
git add .gitattributes model/
git commit -m "Add trained model"
git push
```

## Path A: Hugging Face Spaces (free)

1. Create an account at huggingface.co, verify your email.
2. Click your avatar > New Space. Name: `neuroscan-ng`. License: mit. SDK: **Docker** (blank template). Hardware: CPU basic (free). Visibility: Public. Create.
3. Connect your code. Easiest way, from your local folder:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/neuroscan-ng
git push hf main
```

   When asked for a password, use a Hugging Face access token (Settings > Access Tokens > New token, type Write).
4. In the Space page: Settings > Variables and secrets > New secret. Name: `NEUROSCAN_SECRET`. Value: any long random string (run `python -c "import secrets; print(secrets.token_hex(32))"` to make one).
5. The Space builds automatically from the Dockerfile (5 to 10 minutes the first time). When the status turns green, your app is live at `https://YOUR_HF_USERNAME-neuroscan-ng.hf.space`.
6. Every future `git push hf main` redeploys automatically.

Know this limitation: Space storage is ephemeral. The SQLite database and uploaded scans reset whenever the Space rebuilds or restarts. Fine for demos and your defense, not for keeping real records.

## Path B: DigitalOcean droplet (persistent, professional)

### B1. Get the free credits
1. Apply for the GitHub Student Developer Pack at education.github.com/pack with your YabaTech proof of enrollment (student ID or admission letter). Approval usually takes a few days.
2. In the pack, open the DigitalOcean offer and create your account through that link to receive the credit (typically $200 for 12 months).

### B2. Create the server
1. DigitalOcean dashboard > Create > Droplet.
2. Region: Frankfurt or London (lowest latency to Nigeria). Image: Ubuntu 24.04 LTS. Size: Basic, Regular, **2 GB RAM / 1 vCPU** ($12/mo against your credit; 1 GB is too small for TensorFlow).
3. Authentication: SSH key if you have one, otherwise Password. Create Droplet, note the IP address.

### B3. Install the app
SSH in (`ssh root@YOUR_IP`) and run:

```bash
apt update && apt install -y python3-venv python3-pip nginx git
adduser --system --group neuroscan
cd /opt
git clone https://github.com/YOUR_USERNAME/neuroscan-ng.git
cd neuroscan-ng
python3 -m venv venv
venv/bin/pip install -r requirements.txt tensorflow-cpu gunicorn
chown -R neuroscan:neuroscan /opt/neuroscan-ng
```

### B4. Run it as a service

```bash
cp deploy/neuroscan.service /etc/systemd/system/
nano /etc/systemd/system/neuroscan.service   # replace CHANGE_ME_LONG_RANDOM with a real secret
systemctl daemon-reload
systemctl enable --now neuroscan
systemctl status neuroscan                    # should say active (running)
```

### B5. Put nginx in front

```bash
cp deploy/nginx.conf /etc/nginx/sites-available/neuroscan
nano /etc/nginx/sites-available/neuroscan     # replace YOUR_DOMAIN with your domain or the IP
ln -s /etc/nginx/sites-available/neuroscan /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
```

The app is now live at `http://YOUR_IP`.

### B6. Domain and HTTPS (recommended)
1. Claim the free Namecheap domain from the Student Pack (or any registrar).
2. At the registrar, add an A record pointing `@` (and `www`) to your droplet IP.
3. Back on the server:

```bash
apt install -y certbot python3-certbot-nginx
certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

Certbot edits nginx and sets up auto-renewal. You now have `https://yourdomain.com`.

### B7. Updating later

```bash
cd /opt/neuroscan-ng && git pull && systemctl restart neuroscan
```

## Installing the trained model on either path

After running `train_neuroscan_colab.ipynb`, unzip `neuroscan_model.zip` into `model/`, commit, and push (Path A: push to `hf`; Path B: push to GitHub then B7). The demo banner disappears, Grad-CAM switches on, and the performance page shows your real metrics.

## Quick checklist before showing anyone
- [ ] `NEUROSCAN_SECRET` set to a real random value
- [ ] Register a test account, upload a scan, confirm the result page loads
- [ ] `/performance` shows your trained metrics (or the honest pending state)
- [ ] On Path B: HTTPS working, `systemctl status neuroscan` green after a reboot (`reboot` once to confirm)
