mkdir cameoval_a2m
#format a3m to a2m using hh-suite/scripts/reformat.pl
pth="/scratch/00946/zzhang/data/openfold/cameo/alignments/"
outfile="_uniref90_hits.a2m"
for p in 6tf4_A 6wmk_A 6wqc_A 6xqj_A 6z01_A 7ab3_E 7amc_A 7ar0_B 7atr_A 7au7_A 7awk_A 7b0d_A 7b1k_B 7b1w_F 7b26_C 7b28_F 7b29_A 7b2a_A 7b2o_A 7b3a_A 7b4q_A 7b7t_A 7bbz_A 7bcb_B 7bcj_A 7bcz_A 7bew_A 7bhy_A 7bny_B 7cjs_B 7da9_A 7dah_E 7dck_A 7dcm_A 7dfe_A 7djy_A 7dk9_A 7dkk_A 7dko_A 7dmf_A 7dms_A 7dnm_B 7dnu_A 7don_B 7dq9_A 7dqg_A 7dru_C 7dsu_B 7dtp_A 7dut_A 7duv_A 7dvn_A 7e0m_A 7e2v_A 7e37_A 7e3z_A 7ee3_C 7eea_A 7ef6_A 7eft_B 7eqx_A 7eqx_C 7esx_A 7et8_A 7ezg_A 7f0h_A 7f17_B 7f6e_B 7f8a_A 7fbh_A 7fbp_B 7fe3_A 7fh3_A 7fip_A 7kdx_B 7kgc_A 7kik_A 7kiu_A 7kos_A 7kqv_D 7ksp_A 7kua_A 7kuw_A 7kzh_A 7l19_C 7l6j_A 7l6y_A 7lbu_A 7lc5_A 7lew_B 7ljh_A 7lnu_B 7ls0_B 7lvz_D 7lxc_A 7m0q_B 7m4m_A 7m4n_A 7mfi_A 7mfw_B 7mpz_A 7mqy_A 7mrk_D 7mro_A 7mrq_A 7mrs_B 7msj_A 7mu9_A 7ndr_A 7nf9_A 7nl4_A 7nmb_A 7nwz_D 7nx0_B 7nx0_E 7ny6_A 7o49_F 7o62_B 7obm_A 7ofn_A 7og0_B 7ool_A 7ouq_A 7p1b_C 7p1v_A 7p3b_B 7p3t_B 7p82_C 7pbk_A 7pkx_A 7plb_B 7plq_B 7poh_B 7ppp_A 7pq7_A 7pqf_A 7prd_A 7q03_A 7q1b_A 7q47_A 7r84_A 7rbw_A 7rds_A 7rdt_A 7re4_A 7re6_A 7rg7_A 7rrm_C 7rwk_A 7s02_C 7s13_C 7s13_L 7s6g_A 7s94_C 7sf6_A 7sh3_A 7siq_A 7sir_A 7spo_C 7spp_C 7swh_A 7swk_B 7sy9_A 7t24_A 7t5f_B 7t71_A 7t8o_B 7t9w_B 7tav_B 7tbs_A 7tcb_B 7v1v_A 7v5y_B 7v6p_A 7vdy_A 7vmu_A 7vnb_A 7vpf_A 7w83_A 7wbr_A
do
/work/09123/mjt2211/ls6/tmp/hh-suite/scripts/reformat.pl a3m a2m "$pth$p/uniref90_hits.a3m" "$p$outfile"
done

# make hmmfile out of a2m
mkdir hmmfiles
pth="/work/09123/mjt2211/ls6/cameoval_a2m/"
for p in 6tf4_A 6wmk_A 6wqc_A 6xqj_A 6z01_A 7ab3_E 7amc_A 7ar0_B 7atr_A 7au7_A 7awk_A 7b0d_A 7b1k_B 7b1w_F 7b26_C 7b28_F 7b29_A 7b2a_A 7b2o_A 7b3a_A 7b4q_A 7b7t_A 7bbz_A 7bcb_B 7bcj_A 7bcz_A 7bew_A 7bhy_A 7bny_B 7cjs_B 7da9_A 7dah_E 7dck_A 7dcm_A 7dfe_A 7djy_A 7dk9_A 7dkk_A 7dko_A 7dmf_A 7dms_A 7dnm_B 7dnu_A 7don_B 7dq9_A 7dqg_A 7dru_C 7dsu_B 7dtp_A 7dut_A 7duv_A 7dvn_A 7e0m_A 7e2v_A 7e37_A 7e3z_A 7ee3_C 7eea_A 7ef6_A 7eft_B 7eqx_A 7eqx_C 7esx_A 7et8_A 7ezg_A 7f0h_A 7f17_B 7f6e_B 7f8a_A 7fbh_A 7fbp_B 7fe3_A 7fh3_A 7fip_A 7kdx_B 7kgc_A 7kik_A 7kiu_A 7kos_A 7kqv_D 7ksp_A 7kua_A 7kuw_A 7kzh_A 7l19_C 7l6j_A 7l6y_A 7lbu_A 7lc5_A 7lew_B 7ljh_A 7lnu_B 7ls0_B 7lvz_D 7lxc_A 7m0q_B 7m4m_A 7m4n_A 7mfi_A 7mfw_B 7mpz_A 7mqy_A 7mrk_D 7mro_A 7mrq_A 7mrs_B 7msj_A 7mu9_A 7ndr_A 7nf9_A 7nl4_A 7nmb_A 7nwz_D 7nx0_B 7nx0_E 7ny6_A 7o49_F 7o62_B 7obm_A 7ofn_A 7og0_B 7ool_A 7ouq_A 7p1b_C 7p1v_A 7p3b_B 7p3t_B 7p82_C 7pbk_A 7pkx_A 7plb_B 7plq_B 7poh_B 7ppp_A 7pq7_A 7pqf_A 7prd_A 7q03_A 7q1b_A 7q47_A 7r84_A 7rbw_A 7rds_A 7rdt_A 7re4_A 7re6_A 7rg7_A 7rrm_C 7rwk_A 7s02_C 7s13_C 7s13_L 7s6g_A 7s94_C 7sf6_A 7sh3_A 7siq_A 7sir_A 7spo_C 7spp_C 7swh_A 7swk_B 7sy9_A 7t24_A 7t5f_B 7t71_A 7t8o_B 7t9w_B 7tav_B 7tbs_A 7tcb_B 7v1v_A 7v5y_B 7v6p_A 7vdy_A 7vmu_A 7vnb_A 7vpf_A 7w83_A 7wbr_A
do
bin/hmmbuild --amino --informat a2m "hmmfiles/$p" "$pth$p$outfile"
done

mkdir hmmstats
for p in 6tf4_A 6wmk_A 6wqc_A 6xqj_A 6z01_A 7ab3_E 7amc_A 7ar0_B 7atr_A 7au7_A 7awk_A 7b0d_A 7b1k_B 7b1w_F 7b26_C 7b28_F 7b29_A 7b2a_A 7b2o_A 7b3a_A 7b4q_A 7b7t_A 7bbz_A 7bcb_B 7bcj_A 7bcz_A 7bew_A 7bhy_A 7bny_B 7cjs_B 7da9_A 7dah_E 7dck_A 7dcm_A 7dfe_A 7djy_A 7dk9_A 7dkk_A 7dko_A 7dmf_A 7dms_A 7dnm_B 7dnu_A 7don_B 7dq9_A 7dqg_A 7dru_C 7dsu_B 7dtp_A 7dut_A 7duv_A 7dvn_A 7e0m_A 7e2v_A 7e37_A 7e3z_A 7ee3_C 7eea_A 7ef6_A 7eft_B 7eqx_A 7eqx_C 7esx_A 7et8_A 7ezg_A 7f0h_A 7f17_B 7f6e_B 7f8a_A 7fbh_A 7fbp_B 7fe3_A 7fh3_A 7fip_A 7kdx_B 7kgc_A 7kik_A 7kiu_A 7kos_A 7kqv_D 7ksp_A 7kua_A 7kuw_A 7kzh_A 7l19_C 7l6j_A 7l6y_A 7lbu_A 7lc5_A 7lew_B 7ljh_A 7lnu_B 7ls0_B 7lvz_D 7lxc_A 7m0q_B 7m4m_A 7m4n_A 7mfi_A 7mfw_B 7mpz_A 7mqy_A 7mrk_D 7mro_A 7mrq_A 7mrs_B 7msj_A 7mu9_A 7ndr_A 7nf9_A 7nl4_A 7nmb_A 7nwz_D 7nx0_B 7nx0_E 7ny6_A 7o49_F 7o62_B 7obm_A 7ofn_A 7og0_B 7ool_A 7ouq_A 7p1b_C 7p1v_A 7p3b_B 7p3t_B 7p82_C 7pbk_A 7pkx_A 7plb_B 7plq_B 7poh_B 7ppp_A 7pq7_A 7pqf_A 7prd_A 7q03_A 7q1b_A 7q47_A 7r84_A 7rbw_A 7rds_A 7rdt_A 7re4_A 7re6_A 7rg7_A 7rrm_C 7rwk_A 7s02_C 7s13_C 7s13_L 7s6g_A 7s94_C 7sf6_A 7sh3_A 7siq_A 7sir_A 7spo_C 7spp_C 7swh_A 7swk_B 7sy9_A 7t24_A 7t5f_B 7t71_A 7t8o_B 7t9w_B 7tav_B 7tbs_A 7tcb_B 7v1v_A 7v5y_B 7v6p_A 7vdy_A 7vmu_A 7vnb_A 7vpf_A 7w83_A 7wbr_A
do
bin/hmmstat "hmmfiles/$p" >> "hmmstats/$p"
done


#### HOW TO LOOP THROGH FILE NAMES IN A DIR

for dir in /scratch/09120/sk844/validation_set_cameo/alignments/*
do
p="${dir##*/}"
echo $p
done

# Same run as above but for Sachin

mkdir sk844/validation_set_a2m #location where i will put it all
cd sk844/validation_set_a2m
#format a3m to a2m using hh-suite/scripts/reformat.pl
pth="/scratch/09120/sk844/validation_set_cameo/alignments/"
outfile="_uniref90_hits.a2m"
for dir in /scratch/09120/sk844/validation_set_cameo/alignments/*
do
p="${dir##*/}"
/work/09123/mjt2211/ls6/tmp/hh-suite/scripts/reformat.pl a3m a2m "$pth$p/uniref90_hits.a3m" "$p$outfile"
done


# make hmmfile out of a2m
cd ..
mkdir sk844/hmmfiles
cd ..
pth="/work/09123/mjt2211/ls6/sk844/validation_set_a2m/"
for dir in /scratch/09120/sk844/validation_set_cameo/alignments/*
do
p="${dir##*/}"
bin/hmmbuild --amino --informat a2m "sk844/hmmfiles/$p" "$pth$p$outfile"
done

mkdir sk844/hmmstats
for dir in /scratch/09120/sk844/validation_set_cameo/alignments/*
do
p="${dir##*/}"
bin/hmmstat "sk844/hmmfiles/$p" >> "sk844/hmmstats/$p"
done
#6tf.. 406 3.119217
#7lew_b 48 1.573242