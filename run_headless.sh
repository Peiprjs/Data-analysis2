python -m pip install --upgrade pip
pip install jupyter nbconvert nbformat
pip install -r notebooks/requirements.txt

sudo apt-get update
sudo apt-get install -y texlive-xetex texlive-fonts-recommended texlive-plain-generic pandoc

cd notebooks
jupyter nbconvert --to notebook --execute --inplace data-pipeline.ipynb

jupyter nbconvert --to html data-pipeline.ipynb

jupyter nbconvert --to pdf data-pipeline.ipynb

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

cp data-pipeline.ipynb ../outputs/data-pipeline-${TIMESTAMP}.ipynb
mv data-pipeline.html ../outputs/data-pipeline-${TIMESTAMP}.html
mv data-pipeline.pdf ../outputs/data-pipeline-${TIMESTAMP}.pdf

echo "Done with running!"