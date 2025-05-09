package com.igrs.info;

import android.content.pm.PackageInfo;
import android.content.pm.PackageInstaller;
import android.content.pm.PackageManager;
import android.content.pm.PackageParser;
import android.content.pm.parsing.ApkLiteParseUtils;
import android.content.pm.parsing.PackageLite;
import android.content.pm.parsing.result.ParseResult;
import android.content.pm.parsing.result.ParseTypeImpl;
import android.content.pm.pkg.FrameworkPackageUserState;
import com.android.internal.content.InstallLocationUtils;
import com.igrs.installer.IPackageInfo;
import com.igrs.installer.utils.LogUtil;

import java.io.File;
import java.io.IOException;

public class PackageInfoImpl implements IPackageInfo {
    @Override
    public PackageInfo getPackageInfo(PackageParser.Package parsed) {
        return PackageParser.generatePackageInfo(parsed, null,
                PackageManager.GET_PERMISSIONS, 0, 0, null,
                new PackageUserState());
    }

    @Override
    public void updateSessionParamsInfo(PackageInstaller.SessionParams params, String filePath) {
        File sourceFile = new File(filePath);
        try {
            final ParseTypeImpl input = ParseTypeImpl.forDefaultParsing();
            final ParseResult<android.content.pm.PackageParser.PackageLite> result = ApkLiteParseUtils.parsePackageLite(
                    input.reset(), sourceFile, /* flags */ 0);
            if (result.isError()) {
                LogUtil.e(TAG, "Cannot parse package " + sourceFile + ". Assuming defaults.");
                LogUtil.e(TAG,
                        "Cannot calculate installed size " + sourceFile + ". Try only apk size.");
                params.setSize(sourceFile.length());
            } else {
                final android.content.pm.PackageParser.PackageLite pkg = result.getResult();
                params.setAppPackageName(pkg.getPackageName());
                params.setInstallLocation(pkg.getInstallLocation());
                params.setSize(InstallLocationUtils.calculateInstalledSize(pkg, params.abiOverride));
            }
        } catch (IOException e) {
            e.printStackTrace();
            params.setSize(sourceFile.length());
        }
    }
}