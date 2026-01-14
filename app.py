import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Thêm đường dẫn để import được code từ thư mục src
sys.path.append(os.path.abspath('src'))
from cluster_library import RuleBasedCustomerClusterer

# Cấu hình trang
st.set_page_config(page_title="Customer Clustering Dashboard", layout="wide")

st.title("🛍️ Phân cụm Khách hàng theo Luật Kết hợp")
st.markdown("Dashboard hỗ trợ ra quyết định Marketing dựa trên hành vi mua sắm.")

# --- 1. SIDEBAR: CẤU HÌNH ---
st.sidebar.header("Cấu hình")
k_clusters = st.sidebar.slider("Chọn số lượng cụm (K)", min_value=2, max_value=10, value=3)
top_rules = st.sidebar.number_input("Số lượng luật dùng làm đặc trưng", value=30)
btn_run = st.sidebar.button("🚀 Chạy Phân Cụm")

# --- 2. LOAD DỮ LIỆU ---
# Tự động tìm file trong thư mục data
rules_path = os.path.join('data', 'processed', 'rules.csv')
trans_path = os.path.join('data', 'raw', 'online_retail_II.xlsx')

# Kiểm tra file tồn tại
if not os.path.exists(rules_path) or not os.path.exists(trans_path):
    st.error(f"⚠️ Không tìm thấy dữ liệu! Vui lòng kiểm tra file rules.csv và online_retail_II.xlsx trong thư mục data.")
    st.stop()

@st.cache_data
def load_data(r_path, t_path, k_rules):
    """Hàm load dữ liệu có cache để chạy nhanh hơn"""
    clusterer = RuleBasedCustomerClusterer()
    # Load Rules
    clusterer.load_and_filter_rules(r_path, top_k=k_rules, metric='lift')
    # Build Features (Cache bước này vì nó lâu)
    df_features = clusterer.build_feature_matrix(t_path, mode='binary')
    return clusterer, df_features

# --- 3. XỬ LÝ CHÍNH ---
if btn_run:
    with st.spinner("Đang xử lý dữ liệu... (Có thể mất 1-2 phút lần đầu)"):
        try:
            # Load và xử lý
            clusterer, df_features = load_data(rules_path, trans_path, top_rules)
            
            # Chạy phân cụm
            df_result = clusterer.run_clustering(n_clusters=k_clusters)
            
            st.success("✅ Phân cụm hoàn tất!")
            
            # --- 4. HIỂN THỊ KẾT QUẢ ---
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Biểu đồ Phân tán (PCA)")
                # Vẽ biểu đồ PCA
                fig = plt.figure(figsize=(10, 6))
                clusterer.visualize_clusters() # Hàm này trong library đang plt.show(), cần sửa nhẹ để trả về fig nếu muốn đẹp hơn
                st.pyplot(plt) # Streamlit tự bắt hình vẽ matplotlib
                
            with col2:
                st.subheader("Thống kê Cụm")
                counts = df_result['Cluster'].value_counts().reset_index()
                counts.columns = ['Cụm', 'Số lượng Khách']
                st.dataframe(counts, hide_index=True)
            
            st.divider()
            
            # --- 5. CHI TIẾT INSIGHT ---
            st.subheader(f"🔍 Phân tích chi tiết {k_clusters} nhóm khách hàng")
            
            cluster_profile = df_result.groupby('Cluster').mean()
            
            # Tạo tabs cho từng cụm
            tabs = st.tabs([f"Cụm {i}" for i in range(k_clusters)])
            
            for i, tab in enumerate(tabs):
                with tab:
                    st.markdown(f"**Đặc điểm nổi bật của Cụm {i}:**")
                    # Lấy Top 5 luật phổ biến nhất trong cụm này
                    top_feats = cluster_profile.loc[i].sort_values(ascending=False).head(5)
                    
                    insight_data = []
                    for rule_col, score in top_feats.items():
                        if score > 0: # Chỉ hiện nếu có người mua
                            idx = int(rule_col.split('_')[1])
                            rule_info = clusterer.rules.iloc[idx]
                            insight_data.append({
                                "Xác suất mua": f"{score:.1%}",
                                "Sản phẩm A (Mua cái này)": str(rule_info['antecedents_parsed']),
                                "Sản phẩm B (Mua thêm cái này)": str(rule_info['consequents']),
                                "Lift": round(rule_info['lift'], 2)
                            })
                    
                    if insight_data:
                        st.table(pd.DataFrame(insight_data))
                        st.info("💡 **Gợi ý:** Dựa trên sản phẩm A để bán chéo sản phẩm B cho nhóm khách này.")
                    else:
                        st.warning("Nhóm này chưa có luật nào nổi bật (Khách vãng lai).")

        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")

else:
    st.info("👈 Bấm nút **'Chạy Phân Cụm'** bên thanh menu để bắt đầu phân tích.")