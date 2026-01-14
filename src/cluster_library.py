import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import ast

class RuleBasedCustomerClusterer:
    """
    Class trung tâm để xử lý phân cụm khách hàng dựa trên luật kết hợp.
    """
    def __init__(self):
        self.rules = None
        self.customer_features = None
        self.kmeans_model = None
        self.labels = None
        self.feature_columns = []

    def _parse_frozenset(self, s):
        """Hàm phụ trợ: Xử lý chuỗi 'frozenset({...})' trong CSV thành Python set"""
        try:
            # Nếu chuỗi bắt đầu bằng frozenset, cắt bỏ phần thừa
            if isinstance(s, str) and s.startswith("frozenset"):
                # Cách đơn giản: lấy nội dung bên trong ngoặc {}
                s_clean = s.replace("frozenset({", "").replace("})", "").replace("'", "").replace('"', "")
                if not s_clean: return set()
                return set(item.strip() for item in s_clean.split(','))
            return s
        except:
            return set()

    def load_and_filter_rules(self, rules_path, top_k=50, metric='lift'):
        """
        Đọc file rules.csv và lọc lấy Top-K luật mạnh nhất.
        """
        print(f"🔄 Đang tải luật từ {rules_path}...")
        df_rules = pd.read_csv(rules_path)
        
        # Xử lý cột antecedents (đang là string -> set)
        df_rules['antecedents_parsed'] = df_rules['antecedents'].apply(self._parse_frozenset)
        
        # Sắp xếp và lấy top K
        self.rules = df_rules.sort_values(by=metric, ascending=False).head(top_k).reset_index(drop=True)
        print(f"✅ Đã chọn {len(self.rules)} luật tốt nhất dựa trên {metric}.")
        return self.rules

    def build_feature_matrix(self, transactions_path, mode='binary'):
        """
        Tạo ma trận đặc trưng cho khách hàng.
        """
        print("🔄 Đang xử lý giao dịch để tạo đặc trưng (bước này hơi lâu)...")
        
        # 1. Đọc dữ liệu giao dịch
        if transactions_path.endswith('.xlsx'):
            df = pd.read_excel(transactions_path)
        else:
            df = pd.read_csv(transactions_path)
            
        # ===> ĐOẠN SỬA LỖI (Thêm vào đây) <===
        # Tự động đổi tên cột nếu tên chưa chuẩn (xử lý vụ dấu cách)
        df.rename(columns={
            'Customer ID': 'CustomerID',  # Sửa Customer ID -> CustomerID
            'Price': 'UnitPrice',         # Sửa Price -> UnitPrice (nếu có)
            'Invoice': 'InvoiceNo'        # Sửa Invoice -> InvoiceNo (nếu có)
        }, inplace=True)
        
        # Kiểm tra xem đã có cột CustomerID chưa, nếu chưa thì báo lỗi rõ ràng hơn
        if 'CustomerID' not in df.columns:
            print(f"❌ Lỗi: Không tìm thấy cột 'CustomerID'. Các cột hiện có: {list(df.columns)}")
            return None
        # ======================================

        # Làm sạch cơ bản để chắc chắn có CustomerID
        df = df.dropna(subset=['CustomerID'])
        try:
            df['CustomerID'] = df['CustomerID'].astype(int)
        except:
            pass # Phòng trường hợp ID có chữ cái
        
        # 2. Gom nhóm: Mỗi khách hàng sở hữu tập sản phẩm nào?
        # Output: {12345: {'A', 'B', 'C'}, ...}
        customer_baskets = df.groupby('CustomerID')['Description'].apply(lambda x: set(str(i) for i in x)).to_dict()
        
        print(f"   -> Tìm thấy {len(customer_baskets)} khách hàng.")

        # 3. Quét từng khách hàng qua từng luật
        data = []
        customer_ids = []
        
        self.feature_columns = [f"Rule_{i}" for i in range(len(self.rules))]

        for cust_id, basket_items in customer_baskets.items():
            row = []
            for _, rule in self.rules.iterrows():
                # Lấy tập sản phẩm vế trái của luật (Antecedents)
                rule_items = rule['antecedents_parsed']
                
                # Kiểm tra: Giỏ hàng khách có chứa hết vế trái luật ko?
                if rule_items.issubset(basket_items):
                    if mode == 'binary':
                        row.append(1)
                    else: # weighted
                        row.append(rule['lift'])
                else:
                    row.append(0)
            
            data.append(row)
            customer_ids.append(cust_id)

        # Tạo DataFrame kết quả
        self.customer_features = pd.DataFrame(data, columns=self.feature_columns, index=customer_ids)
        print(f"✅ Ma trận đặc trưng hoàn tất: {self.customer_features.shape}")
        return self.customer_features
    def find_optimal_k(self, max_k=10):
        """Vẽ biểu đồ Elbow và Silhouette để gợi ý số cụm K"""
        if self.customer_features is None:
            print("❌ Chưa có dữ liệu đặc trưng!")
            return

        print("🔄 Đang chạy thử nghiệm tìm K tối ưu...")
        distortions = []
        sil_scores = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.customer_features)
            distortions.append(kmeans.inertia_)
            sil_scores.append(silhouette_score(self.customer_features, labels))

        # Vẽ hình
        fig, ax1 = plt.subplots(figsize=(12, 5))

        ax1.set_xlabel('Số lượng cụm (k)')
        ax1.set_ylabel('Inertia (Elbow)', color='tab:blue')
        ax1.plot(K_range, distortions, 'bx-')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Silhouette Score', color='tab:red')
        ax2.plot(K_range, sil_scores, 'ro--')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title('Phương pháp Elbow và Silhouette')
        plt.show()

    def run_clustering(self, n_clusters=3):
        """Chạy K-Means chính thức"""
        print(f"🚀 Đang phân cụm với k={n_clusters}...")
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans_model.fit_predict(self.customer_features)
        
        # Gán nhãn vào DataFrame gốc để phân tích
        result = self.customer_features.copy()
        result['Cluster'] = self.labels
        return result

    def visualize_clusters(self):
        """Vẽ biểu đồ phân tán 2D (PCA)"""
        pca = PCA(n_components=2)
        components = pca.fit_transform(self.customer_features)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=components[:,0], y=components[:,1], hue=self.labels, palette='viridis', s=80)
        plt.title('Biểu đồ phân cụm khách hàng (PCA 2D)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(title='Cluster')
        plt.show()