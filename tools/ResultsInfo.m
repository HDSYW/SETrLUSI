function ResultsInfo(result,Para)

    if Para. AutoRec == "ON"
        FilePth = sprintf('./AutoResult/Results_%s_%s.txt',Para.name,Para.time);
        fid = fopen(FilePth, 'a+'); 
        num=length(fieldnames(result));
        if num==8
            if result.chongfu==1
                fprintf(fid,'%s\n',result.Mod);
            end
            fprintf(fid,'|%.0f|%.2f|%.2f|%.2f|%.2f|\n',result.seed,result.ac_test,result.F, result.GM,Para.runtime);
            status=fclose(fid);
        end
        if num==9
            if result.chongfu==1
                fprintf(fid,'%s\n',result.Mod);
            end
            fprintf(fid,'|%.0f|%.2f|%.2f|%.2f|%.2f|\n',result.seed,result.ac_test,result.F, result.GM,Para.runtime);
            status=fclose(fid);
        end
        if num==10
            if result.chongfu==1
                fprintf(fid,'%s\n',result.Mod);
            end
            fprintf(fid,'|%.0f|%.2f|%.2f|%.2f|%.2f|\n',result.seed,result.ac_test,result.F, result.GM,Para.runtime);
            status=fclose(fid);
        end
        if num==11
            if result.chongfu==1
                fprintf(fid,'%s\n',result.Mod);
            end
            fprintf(fid,'|%.0f|%.2f|%.2f|%.2f|%.0f|%.2f|\n',result.seed,result.ac_test,result.F, result.GM, result.lam,Para.runtime);
            status=fclose(fid);
        end
        if num==12
            if result.chongfu==1
                fprintf(fid,'%s\n',result.Mod);
            end
            fprintf(fid,'|%.0f|%.2f|%.2f|%.2f|%.0f|%.1f|%.2f|\n',result.seed,result.ac_test,result.F, result.GM, result.lam,result.p3,Para.runtime);
            status=fclose(fid);
        end
        if num==13
            if result.chongfu==1
                fprintf(fid,'%s\n',result.Mod);
            end
            fprintf(fid,'|%.0f|%.2f|%.2f|%.2f|%.0f|%.0f|%.0f|%.2f|\n',result.seed,result.ac_test,result.F, result.GM,log2(result.lam),log2(result.kp1),log2(result.v_sig),log2(result.p4),Para.runtime);
            status=fclose(fid);
        end        
        if num==14
            if result.chongfu==1
                fprintf(fid,'%s \n',result.Mod);
            end
            fprintf(fid,'|%.0f|%.2f|%.2f|%.2f|%.0f|%.0f|%.0f|%.2f|%.2f|\n',result.seed,result.ac_test,result.F, result.GM,log2(result.kp1),log2(result.lam),log2(result.v_sig),result.tao_1,Para.runtime);
            status=fclose(fid);
        end
    end
end