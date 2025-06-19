## **Spécifications de la machine**

### **Système d'exploitation**
- **Ubuntu 24.04.2 LTS** (GNU/Linux 6.11.0-26-generic x86_64)
- Architecture 64-bit

### **GPU - Excellente pour l'IA !**
- **NVIDIA Quadro RTX 5000**
- **16 GB de VRAM** (16384 MiB)
- Driver NVIDIA 550.144.03
- **CUDA 12.4** disponible
- **64 GB de RAM** (65,509,388 KB total)

### **Stockage**
- Disque principal : **465.8 GB** (SSD probablement sur /dev/sdc)
- Deux disques additionnels de **953.9 GB** chacun (sda et sdb)
- Accès à un **NAS partagé** (nas1-home et nas1-yatoo)

### **Réseau**
- IP interne : **10.10.9.62**
- Accès SSH configuré

## **Comment bien travailler avec cette machine**

### **1. Connexion et accès**
```bash
# Depuis ta machine locale (pc-eii210)
ssh -X asidimoh@10.10.9.62
```

### **2. Configuration recommandée**

**Générer des clés SSH pour éviter de taper le mot de passe :**
```bash
# Sur ta machine locale
ssh-keygen -t rsa -b 4096
ssh-copy-id asidimoh@10.10.9.62
```

### **3. Utilisation optimale des ressources**

**Monitoring des ressources :**
```bash
htop          # CPU et RAM
nvidia-smi    # GPU
nvtop         # GPU en temps réel (plus joli)
df -h         # Espace disque
```

---

## **Pour VS Code - PAS besoin de builder/transférer !**

### **VS Code Remote SSH (RECOMMANDÉE)**

**Installation :**
1. Installe l'extension **"Remote - SSH"** dans VS Code
2. Configure la connexion SSH

**Configuration dans VS Code :**
```bash
# Ctrl+Shift+P → "Remote-SSH: Connect to Host"
# Ou configure dans ~/.ssh/config sur ta machine locale :

Host gpu-server
    HostName 10.10.9.62
    User asidimoh
    ForwardX11 yes
```

### **Accede a l'interface sur la machine distante**
```bash
# Sur ta machine locale
ssh -L 8000:localhost:8000 asidimoh@10.10.9.62
# Puis ouvre http://localhost:8000 dans ton navigateur
```