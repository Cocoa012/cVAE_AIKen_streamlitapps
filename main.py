import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル定義
class CVAE(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # エンコーダネットワーク
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, 14, 14]
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # [batch, 64, 7, 7]
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.LeakyReLU(0.2),
        )
        
        # クラス埋め込み
        self.class_embedding = nn.Embedding(num_classes, 512)
        
        # 潜在変数パラメータ
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        
        # デコーダネットワーク
        self.decoder_input = nn.Linear(latent_dim + num_classes, 512)
        self.decoder = nn.Sequential(
            nn.Linear(512, 64 * 7 * 7),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # [batch, 32, 14, 14]
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # [batch, 1, 28, 28]
            nn.Sigmoid()
        )
    
    def encode(self, x, y):
        x_encoded = self.encoder(x)
        y_embedded = self.class_embedding(y)
        concat_input = torch.cat([x_encoded, y_embedded], dim=1)
        
        mu = self.fc_mu(concat_input)
        logvar = self.fc_logvar(concat_input)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y):
        # クラスのワンホット表現を作成
        y_onehot = F.one_hot(y, self.num_classes).float()
        
        # zとy_onehotを連結
        z_and_y = torch.cat([z, y_onehot], dim=1)
        
        # デコード
        h = F.leaky_relu(self.decoder_input(z_and_y), 0.2)
        return self.decoder(h)
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar
    
    def generate(self, z, y):
        """与えられた潜在変数zとクラスラベルyから画像を生成"""
        return self.decode(z, y)

# ページ設定
st.set_page_config(
    page_title="CVAE 数字生成アプリ",
    page_icon="🧠",
    layout="wide",
)

# CSSでUIをカスタマイズ
st.markdown("""
<style>
    /* 全体のテーマを暗くする */
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    
    /* ヘッダーのスタイル */
    h1, h2, h3 {
        color: #FFFFFF !important;
    }
    
    /* スライダーのスタイル */
    .stSlider > div > div {
        background-color: #3D3D3D !important;
    }
    
    /* ボタンのスタイル - 立体感を出す */
    .stButton > button {
        background-color: #4A4A4A;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #686868;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        background-color: #383838;
        box-shadow: 0 2px 4px 0 rgba(0,0,0,0.1);
        transform: translateY(1px);
    }
    
    /* セレクトボックスのスタイル */
    .stSelectbox > div > div {
        background-color: #3D3D3D;
        color: white;
    }
    
    /* 分割線のスタイル */
    hr {
        border: 1px solid #3D3D3D;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("CVAE 数字生成アプリ")
    st.write("条件付き変分オートエンコーダ(CVAE)で手書き数字を生成します")
    
    # モデルのロード
    @st.cache_resource
    def load_model():
        model = CVAE(latent_dim=3, num_classes=10).to(device)
        model.load_state_dict(torch.load("cvae.pth", map_location=device))
        model.eval()
        return model
    
    try:
        model = load_model()
        st.success("✅ モデルのロードに成功しました")
    except Exception as e:
        st.error(f"❌ モデルのロードに失敗しました: {e}")
        st.info("モデルファイル 'cvae.pth' が同じディレクトリに存在するか確認してください")
        return
    
    st.subheader("数字生成のパラメータ設定")
    
    # 生成する数字の選択
    digit = st.selectbox("生成する数字を選択", list(range(10)))
    
    # 潜在空間のパラメータ
    st.write("潜在変数のパラメータ調整")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        z1 = st.slider("z1", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
    
    with col2:
        z2 = st.slider("z2", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
    
    with col3:
        z3 = st.slider("z3", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
    
    # ランダムなzベクトルを生成するボタン
    if st.button("🎲 ランダムなパラメータを生成"):
        z1 = np.random.normal(0, 1)
        z2 = np.random.normal(0, 1)
        z3 = np.random.normal(0, 1)
        st.session_state.z1 = z1
        st.session_state.z2 = z2
        st.session_state.z3 = z3
        st.experimental_rerun()
    
    # 生成ボタン
    if st.button("✨ 数字を生成"):
        # 潜在変数ベクトルの作成
        z = torch.tensor([[z1, z2, z3]], dtype=torch.float32).to(device)
        
        # クラスラベルの作成
        label = torch.tensor([digit], dtype=torch.long).to(device)
        
        # 画像生成
        with torch.no_grad():
            x_gen = model.generate(z, label)
        
        # 画像の表示
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(x_gen.squeeze().cpu().numpy(), cmap='gray_r')
        ax.axis('off')
        ax.set_title(f"生成された数字: {digit}", fontsize=16)
        
        # 黒背景のプロット設定
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#121212')
        ax.title.set_color('white')
        
        st.pyplot(fig)
        
        # 生成されたパラメータ情報の表示
        st.subheader("生成パラメータ情報")
        st.code(f"数字: {digit}\nz1: {z1:.4f}\nz2: {z2:.4f}\nz3: {z3:.4f}")
    
    st.markdown("---")
    st.markdown("### 使い方")
    st.markdown("""
    1. 生成したい数字を選択してください（0〜9）
    2. 潜在変数（z1, z2, z3）を調整して異なるスタイルの数字を生成できます
    3. ランダムな潜在変数でも試してみてください
    4. 「数字を生成」ボタンをクリックすると生成された数字が表示されます
    """)

if __name__ == "__main__":
    main()
