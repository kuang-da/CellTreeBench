mkdir -p
rsync -avz --progress \
--exclude="phylodist/outs" \
--exclude="simulation" \
--exclude="simulation/bmtm-pipeline/outs" \
--exclude="simulation/bmtm-pipeline/out" \
-e "ssh -i /mnt/sda/0-projects/1-phydist/main/phylodist/scripts/guanwenec2.pem" \
/mnt/sda/0-projects/1-phydist/main/ ec2-user@18.118.18.142:/home/ec2-user/1-phydist/main/


mkdir -p
rsync -avz --progress \
-e "ssh -i /mnt/sda/0-projects/1-phydist/main/phylodist/scripts/guanwenec2.pem" \
--exclude="simulation/bmtm-pipeline/outs" \
--exclude="simulation/bmtm-pipeline/out" \
/mnt/sda/0-projects/1-phydist/main/simulation/bmtm-pipeline ec2-user@18.118.18.142:/home/ec2-user/1-phydist/main/simulation/bmtm-pipeline
